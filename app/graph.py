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


class InspectionState(TypedDict):
    image_path: str
    detections: list[Detection]
    defect_analysis: dict


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


# ============================================================
# 4. DEFECT ANALYSIS AGENT
# ============================================================

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


# ============================================================
# 5. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(InspectionState)


# Register nodes
builder.add_node(
    "vision",
    vision_node,
)

builder.add_node(
    "defect_analysis",
    defect_analysis_node,
)


# Define flow
builder.add_edge(
    START,
    "vision",
)

builder.add_edge(
    "vision",
    "defect_analysis",
)

builder.add_edge(
    "defect_analysis",
    END,
)


# Compile graph
graph = builder.compile()