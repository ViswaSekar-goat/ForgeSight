CREATE DATABASE IF NOT EXISTS forgesight;

USE forgesight;

CREATE TABLE inspections (
    inspection_id CHAR(36) PRIMARY KEY , 
    sample_id VARCHAR(100) NOT NULL , 
    image_path VARCHAR(500) NOT NULL , 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP , 
    final_action VARCHAR(30)
);

CREATE TABLE detections (

    detection_id BIGINT AUTO_INCREMENT PRIMARY KEY , 
    inspection_id CHAR(36) NOT NULL , 
    class_name VARCHAR(100) NOT NULL,

    confidence FLOAT NOT NULL,

    x1 FLOAT NOT NULL,
    y1 FLOAT NOT NULL,
    x2 FLOAT NOT NULL,
    y2 FLOAT NOT NULL,

    FOREIGN KEY (inspection_id)
        REFERENCES inspections(inspection_id)
);