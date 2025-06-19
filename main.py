import os
import json
import time
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
import ast

load_dotenv()
# es_dashboard_username = os.getenv("ES_DASHBOARD_USERNAME")
# es_dashboard_password = os.getenv("ES_DASHBOARD_PASSWORD")
# es_dashboard_hosts = os.getenv("ES_DASHBOARD_HOSTS")
OpenAI_api_key = os.getenv("OPEN_AI_KEY_IC")

# es_dashboard = Elasticsearch(ast.literal_eval(es_dashboard_hosts),
#                             basic_auth=(es_dashboard_username, es_dashboard_password), verify_certs=False,
#                             request_timeout=300)

client = OpenAI(api_key=OpenAI_api_key)


class ValidationConfig:
    def __init__(
        self,
        task_name: str,
        input_fields: List[str],
        output_schema: Dict[str, tuple],
        system_prompt: str,
        model_name: str = "gpt-4.1-mini",
        batch_size: int = 100
    ):
        """
        Configure the validation task
        
        Args:
            task_name: Name of the validation task (e.g., "gender_detection", "location_extraction")
            input_fields: List of CSV columns to use as input
            output_schema: Dictionary defining the output fields and their types/descriptions
            system_prompt: The system prompt for the AI
            model_name: OpenAI model to use
            batch_size: Number of items to process in each batch
        """
        self.task_name = task_name
        self.input_fields = input_fields
        self.output_schema = output_schema
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.batch_size = batch_size

        # Dynamically create Pydantic model for validation
        self.validation_model = create_model(
            f"{task_name.title()}ValidationModel",
            **{field: (type_, Field(description=desc)) 
               for field, (type_, desc) in output_schema.items()}
        )

class BulkValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.client = OpenAI(api_key=os.getenv("OPEN_AI_KEY_IC"))
        
    def create_prompt(self, row: pd.Series) -> str:
        """Create a prompt using configured input fields"""
        prompt_parts = []
        
        for field in self.config.input_fields:
            if pd.notna(row[field]):
                prompt_parts.append(f"{field}: {row[field]}")
        
        return "\n".join(prompt_parts)

    def process_batch(self, df: pd.DataFrame, output_path: str):
        """Process a batch of data"""
        batch_requests = []
        
        for idx, row in df.iterrows():
            prompt = self.create_prompt(row)
            cid = f"row_{idx}"
            
            batch_requests.append({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.config.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": self.config.system_prompt
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"}
                }
            })

        # Create batch job and process results
        jsonl_path = Path(f"{self.config.task_name}_batch_input.jsonl")
        try:
            jsonl_path.write_text("\n".join(json.dumps(r) for r in batch_requests))
            
            with jsonl_path.open("rb") as fh:
                batch_file = self.client.files.create(file=fh, purpose="batch")
            
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # Poll for completion
            while True:
                status = self.client.batches.retrieve(batch_job.id)
                if status.status == "completed":
                    break
                elif status.status == "in_progress":
                    print(f"Batch {batch_job.id} is in progress")
                time.sleep(15)
            
            # Process results
            result_bytes = self.client.files.content(status.output_file_id).content
            rows_raw = result_bytes.decode("utf-8").strip().split("\n")
            
            for line in rows_raw:
                obj = json.loads(line)
                cid = obj["custom_id"]
                content = json.loads(obj["response"]["body"]["choices"][0]["message"]["content"])
                
                idx = int(cid.split("_")[1])
                for field, value in content.items():
                    df.at[idx, f"ai_{field}"] = value
            
            df.to_csv(output_path, index=False)
            
        finally:
            if jsonl_path.exists():
                jsonl_path.unlink()



def process_validation_task(
    config: ValidationConfig,
    input_path: str,
    output_path: str,
    total_records_validate: int,
):
    """Main function to process the validation task"""
    df = pd.read_csv(input_path)[:total_records_validate]
    validator = BulkValidator(config)
    
    # Process in batches
    for i in range(0, len(df), config.batch_size):
        batch_df = df.iloc[i:i + config.batch_size]
        validator.process_batch(batch_df, output_path)
        print(f"Processed batch {i//config.batch_size + 1}")



if __name__ == "__main__":
    gender_config = ValidationConfig(
    task_name="gender_detection",
    input_fields=["biography", "captions", "full_name"],
    output_schema={
        "gender": (str, "The predicted gender: 'male' or 'female'"),
        "reasoning": (str, "The reasoning behind the prediction")
    },
    system_prompt="""You are a gender prediction expert. Analyze the provided information 
    and predict the person's gender. Return a JSON object with gender and confidence score.
    Only use 'male' or 'female' as gender values.""",
    model_name="gpt-4.1-mini",
    batch_size=5
    )

    # Process the task
    process_validation_task(
        config=gender_config,
        input_path="C:/Users/vladi/Downloads/Validation_team_data_IG.csv",
        output_path="C:/Users/vladi/Downloads/Validation_team_data_IG_ai_template_gender.csv",
        total_records_validate=15
    )
