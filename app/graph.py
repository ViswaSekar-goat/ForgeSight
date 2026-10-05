from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()



# ============================================================
# 1. LOAD MODELS
# ============================================================

# Our fine-tuned industrial defect detector
model = YOLO("models/forgeSight_yolo.pt")


# Groq LLM used by the Defect Analysis Agent
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key= os.getenv("GROQ_API_KEY") ,
    temperature=0,
)


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

class Detection(TypedDict):
    class_name: str
    confidence: float
    bbox: list[float]


class DefectAnalysis(BaseModel):
    observed_facts: list[str] = Field(
        description=(
            "Facts directly supported by the YOLO detections. "
            "Do not invent visual information."
        )
    )

    assessment: str = Field(
        description=(
            "Interpretation of what the YOLO detections suggest."
        )
    )

    uncertainty: str = Field(
        description=(
            "Describe uncertainty, especially when detection "
            "confidence is low or evidence is ambiguous."
        )
    )

    recommended_next_step: str = Field(
        description=(
            "What should be inspected or investigated next. "
            "Do not make a final accept/reject decision."
        )
    )

class QualityAssessment(BaseModel):
    severity: str = Field(
        description="Potential severity: LOW, MEDIUM, HIGH, or CRITICAL"
    )

    reasoning: str = Field(
        description="Explain why this severity is suggested"
    )

    confidence_level: str = Field(
        description="HIGH, MEDIUM, or LOW confidence in the assessment"
    )

    recommended_action: str = Field(
        description="Suggested workflow action, not an authoritative final decision"
    )

class DecisionRecommendation(BaseModel):
    decision: str = Field(
        description=(
            "Recommended action: ACCEPT, REJECT, REINSPECT, "
            "or HUMAN_REVIEW"
        )
    )

    reasoning: str = Field(
        description="Explain how the evidence supports the recommendation"
    )

    conflicting_evidence: str = Field(
        description="Mention any disagreement or uncertainty between the agents"
    )

    confidence: str = Field(
        description="HIGH, MEDIUM, or LOW confidence in this recommendation"
    )

class InspectionState(TypedDict):
    image_path: str
    detections: list[Detection]
    inspection_history: list[dict]
    defect_analysis: dict
    quality_assessment: dict
    decision: dict
    policy_decision: dict


# ============================================================
# 3. VISION NODE
# ============================================================

def vision_node(state: InspectionState):
    """
    Run YOLO on the input image and convert its predictions
    into structured information that the LangGraph state can use.
    """

    results = model(
        state["image_path"],
        verbose=False,
        device=0,
    )

    detections: list[Detection] = []

    for result in results:

        if result.boxes is None:
            continue

        classes = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
        boxes = result.boxes.xyxy.tolist()

        for cls, conf, box in zip(
            classes,
            confidences,
            boxes,
        ):

            detections.append(
                {
                    "class_name": result.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": [float(value) for value in box],
                }
            )

    return {
        "detections": detections
    }


# ----------------------------- DEFECT NODE ------------------------------------

defect_analyzer = llm.with_structured_output(
    DefectAnalysis
)



def defect_analysis_node(state: InspectionState):
    """
    Analyze YOLO's evidence using the Groq LLM.

    The LLM interprets the evidence but does NOT have authority
    to accept or reject the component.
    """

    prompt = f"""
You are the Defect Analysis Agent in ForgeSight,
an industrial visual inspection system.

A YOLO computer vision model inspected a component and produced
the following detections:

{state["detections"]}

Your responsibilities:

1. Identify the defects that YOLO detected.
2. Treat YOLO detections as observations/evidence.
3. Consider the confidence scores when discussing uncertainty.
4. Distinguish clearly between observed facts and interpretation.
5. Do NOT invent visual information that is not present in the
   supplied detections.
6. Do NOT claim that a defect has a particular root cause.
7. Do NOT make the final ACCEPT or REJECT decision.
8. If the evidence is uncertain, recommend further inspection.
9. Keep the analysis concise and technically defensible.

Remember:

YOLO = perception
You = evidence interpretation
Policy engine = final authority
Human reviewer = safety override
"""

    analysis = defect_analyzer.invoke(prompt)

    return {
        "defect_analysis": analysis.model_dump()
    }

# ----------------------------- QUALITY NODE ------------------------------------

quality_assessor = llm.with_structured_output(QualityAssessment)

def quality_assessment_node(state: InspectionState):

    prompt = f"""
You are the Quality Assessment Agent in ForgeSight,
an industrial visual inspection system.

YOLO detections:
{state["detections"]}

Assess the potential quality severity.

Rules:

1. Use only the supplied evidence.
2. Do not invent measurements or visual information.
3. Consider detection confidence.
4. Use one of:
   LOW
   MEDIUM
   HIGH
   CRITICAL
5. Clearly distinguish severity from certainty.
6. Do not make an authoritative ACCEPT or REJECT decision.
7. If evidence is insufficient, recommend HUMAN_REVIEW or REINSPECTION.
"""

    assessment = quality_assessor.invoke(prompt)

    return {
        "quality_assessment": assessment.model_dump()
    }

# ----------------------------- DECISION NODE ------------------------------------
decision_maker = llm.with_structured_output(
    DecisionRecommendation
)

def decision_node(state: InspectionState):

    prompt = f"""
You are the Decision Agent in ForgeSight,
an industrial visual inspection system.

YOLO detections:
{state["detections"]}

Defect Analysis Agent:
{state["defect_analysis"]}

Quality Assessment Agent:
{state["quality_assessment"]}

Your job is to synthesize the available evidence.

Possible recommendations:

ACCEPT
REJECT
REINSPECT
HUMAN_REVIEW

Rules:

1. Consider all available evidence.
2. Do not invent information.
3. Pay attention to YOLO confidence.
4. Pay attention to uncertainty in the Defect Analysis.
5. Pay attention to severity and confidence in the Quality Assessment.
6. If the evidence is insufficient or conflicting, prefer
   REINSPECT or HUMAN_REVIEW.
7. This is a recommendation only.
8. Do not claim that your recommendation is the authoritative
   final production decision.
"""

    recommendation = decision_maker.invoke(prompt)

    return {
        "decision": recommendation.model_dump()
    }

# ----------------------------- POLICY NODE ------------------------------------


LOW_CONFIDENCE = 0.50
HIGH_CONFIDENCE = 0.75


def policy_engine_node(state: InspectionState):

    detections = state["detections"]

    # No defects detected
    if not detections:

        return {
            "policy_decision": {
                "action": "ACCEPT",
                "reason": "No defects were detected by the vision model.",
                "authority": "deterministic_policy",
            }
        }

    confidences = [
        detection["confidence"]
        for detection in detections
    ]

    # At least one uncertain detection
    if any(confidence < LOW_CONFIDENCE for confidence in confidences):

        return {
            "policy_decision": {
                "action": "REINSPECT",
                "reason": "At least one detected defect has low confidence.",
                "authority": "deterministic_policy",
            }
        }

    # Strong visual evidence
    if any(confidence >= HIGH_CONFIDENCE for confidence in confidences):

        return {
            "policy_decision": {
                "action": "REJECT",
                "reason": "At least one defect has high-confidence visual evidence.",
                "authority": "deterministic_policy",
            }
        }

    # Ambiguous middle region
    return {
        "policy_decision": {
            "action": "HUMAN_REVIEW",
            "reason": "Defect evidence is present but does not meet a deterministic threshold.",
            "authority": "deterministic_policy",
        }
    }


# ============================================================
# 5. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(InspectionState)


builder.add_node("vision", vision_node)
builder.add_node("defect_analysis", defect_analysis_node)
builder.add_node("quality_assessment", quality_assessment_node)
builder.add_node("decision", decision_node)
builder.add_node("policy_engine",policy_engine_node,
)

builder.add_edge(START, "vision")

builder.add_edge("vision", "defect_analysis")
builder.add_edge("vision", "quality_assessment")

builder.add_edge("defect_analysis", "decision")
builder.add_edge("quality_assessment", "decision")

builder.add_edge("decision", "policy_engine")

builder.add_edge("policy_engine", END)

graph = builder.compile()


# Compile graph
graph = builder.compile()