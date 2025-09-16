import json
from pydantic import BaseModel, Field
from typing import List, Optional
import boto3

class LineItem(BaseModel):
    description: Optional[str] = Field(
        None, description="A brief description of the product or service provided."
    )
    quantity: Optional[float] = Field(
        None, description="The number of units of the product or service."
    )
    unit_amount: Optional[float] = Field(
        None, description="The price amount per unit of the product or service."
    )
    total_amount: Optional[float] = Field(
        None, description="The total amount price for the line item, calculated as quantity × unit price."
    )
    
class InvoiceData(BaseModel):
    invoice_number: Optional[str] = Field(
        None, description="The unique identifier or reference number of the invoice or receipt"
    )
    invoice_date: Optional[str] = Field(
        None, description="The date when the invoice/receipt was issued."
    )
    vendor_name: Optional[str] = Field(
        None, description="The name of the company or individual issuing the invoice/receipt."
    )
    line_items: Optional[List[LineItem]] = Field(
        None, description="A list of items described in the invoice."
    )
    subtotal: Optional[float] = Field(
        None, description="The sum of all line item totals before taxes or additional fees."
    )
    tax: Optional[float] = Field(
        None, description="The tax amount applied to the subtotal."
    )
    total_amount: Optional[float] = Field(
        None, description="The final total to be paid including subtotal and taxes."
    )
    currency: Optional[str] = Field(
        None, description="The currency in which the invoice is issued (e.g., USD, EUR, GBP)."
    )

prompt = f"""
You are an intelligent OCR extraction agent capable of understanding and processing documents in multiple languages.
Given an image of an invoice or receipt, extract all relevant information in structured JSON format.
The JSON object must use the schema: {json.dumps(InvoiceData.model_json_schema(), indent=2)}
If any field cannot be found in the invoice, return it as null. Focus on clarity and accuracy, and ignore irrelevant text such as watermarks, headers, or decorative elements. Return the final result strictly in JSON format.
"""
boto_session = boto3.session.Session()
bedrock_client = boto_session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)
model = 'us.meta.llama4-maverick-17b-instruct-v1:0'
#model = 'us.meta.llama4-scout-17b-instruct-v1:0'

def make_multi_images_messages(question, image_paths):
    img_msg = []
    try:
        
        for img in images_path:
            with open(img, "rb") as image_file:
                image_bytes = image_file.read()
            img_1 = Image.open(img)
            imgformat = img_1.format
            imgformat = imgformat.lower()
            img_msg.append({"image": {
                "format": imgformat,
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                })
    except FileNotFoundError:
        print(f"Image file not found at {image_path}")
        image_data = None
        image_media_type = None
    messages = [            
            {
                "role": "user",
                "content": [
                {                        
                    "text": question
                },
                *img_msg
                ]
            }
        ]
    return messages


