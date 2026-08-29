"""
Main runner for the semantic enrichment pipeline.
"""

import os
import json
import logging
from pathlib import Path
from tqdm import tqdm

from .config import OCR_OUTPUT_DIR, SEMANTIC_OUTPUT_DIR, NORMALIZED_DIR, ENHANCED_OUTPUT_DIR, SEMANTIC_EVAL_DIR
from .normalization.normalizer import SemanticNormalizer
from .ner.extractor import BiomedicalNER
from .ontology_mapping.mapper import OntologyMapper
from .benchmarking.evaluator import SemanticEvaluator
from .visualization.analytics import generate_visual_reports
from .schema import EnrichedPrescription

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/semantic_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("semantic_pipeline")

def run_pipeline():
    logger.info("Starting Semantic Enrichment Pipeline...")
    
    normalizer = SemanticNormalizer()
    ner = BiomedicalNER()
    mapper = OntologyMapper()
    
    # 1. Discover all OCR engine output folders
    if not OCR_OUTPUT_DIR.exists():
        logger.error(f"OCR output directory not found: {OCR_OUTPUT_DIR}")
        return

    engine_folders = [f for f in OCR_OUTPUT_DIR.iterdir() if f.is_dir()]
    
    for engine_folder in engine_folders:
        engine_name = engine_folder.name
        logger.info(f"Processing outputs for engine: {engine_name}")
        
        # Create corresponding output folders
        (NORMALIZED_DIR / engine_name).mkdir(parents=True, exist_ok=True)
        (ENHANCED_OUTPUT_DIR / engine_name).mkdir(parents=True, exist_ok=True)
        
        json_files = list(engine_folder.glob("*.json"))
        
        for json_file in tqdm(json_files, desc=f"Engine {engine_name}"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                # Step 1: Normalize
                normalized_data = normalizer.process_ocr_output(ocr_data)
                
                # Step 2: NER
                ner_data = ner.enrich_json(normalized_data)
                
                # Step 3: Ontology Mapping
                enriched_data = mapper.enrich_json(ner_data)
                
                # Save enriched JSON
                output_path = ENHANCED_OUTPUT_DIR / engine_name / json_file.name
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, indent=2)
                
                # Also save a simplified 'normalized' version
                norm_output_path = NORMALIZED_DIR / engine_name / json_file.name
                with open(norm_output_path, 'w', encoding='utf-8') as f:
                    json.dump(normalized_data, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Failed to process {json_file.name}: {e}")

    logger.info("Processing complete. You can now run evaluation using --evaluate.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic Enrichment Pipeline")
    parser.add_argument("--process", action="store_true", default=True, help="Run normalization and enrichment")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmarking evaluation")
    parser.add_argument("--no-process", action="store_false", dest="process", help="Skip processing")
    
    args = parser.parse_args()
    
    # Ensure logs dir exists
    Path("logs").mkdir(exist_ok=True)
    
    if args.evaluate:
        # Step 4: Evaluate only
        logger.info("Starting Evaluation only...")
        evaluator = SemanticEvaluator()
        report = evaluator.run_all()
        evaluator.save_reports(report)
        
        # Step 5: Visual Analytics
        logger.info("Generating visual reports...")
        generate_visual_reports()
        logger.info(f"Evaluation complete. Reports saved to {SEMANTIC_EVAL_DIR}")
    elif args.process:
        run_pipeline()

if __name__ == "__main__":
    main()
