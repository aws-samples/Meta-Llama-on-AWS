import os
import re
import json
from strands import Agent, tool
import boto3
from strands.models import BedrockModel
from pydantic import BaseModel, Field
from typing import Optional, List
import ast



@tool
def convert_currency_to_usd(data):
    try:
        data = ast.literal_eval(data)
        # Validate the currency of each line item carefully and convert to USD if needed.
        mod_data = []
        for i in data:
            if isinstance(i, list):
                for m in i:
                    mod_data.append(m)
            else:
                mod_data.append(i)
        print("***** Currency conversion - Started *****")
        result = []
        items = []
        for item in mod_data:
            # Check if item is a dictionary
            # Check if item has a 'receipts' key with a list value
            if 'receipts' in item:
                # Extract each receipt from the receipts list
                for receipt in item['receipts']:
                    if isinstance(receipt, dict):
                        items.append(receipt)
            else:
                items.append(item)
        #print(items)        
        for item in items:
            result.append(process_receipt(item))
        print("***** Currency conversion - Completed *****")    
        return result
    except Exception as e:
        return f"Error processing currency conversion: {str(e)}"
    
@tool
def cur_converter(amount: float, currency: str) -> float:
    # Exchange rates (could be fetched from an API in production)
    conversion_rates = {
        'GBP': 1.35,
        'EUR': 1.10,
        'JPY': 0.0069,
        # Add other currencies as needed
    }
    
    if currency == 'USD':
        return round(amount, 2)
    elif currency in conversion_rates:
        return round(amount * conversion_rates[currency], 2)
    else:
        # Default to returning the original amount if currency not recognized
        print(f"Warning: No conversion rate found for {currency}")
        return amount
@tool
def extract_float(value):
    """Extract float value from string regardless of currency symbol"""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        # Remove currency symbols and commas
        cleaned = value.replace('$', '').replace('£', '').replace('€', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None

@tool
def process_receipt(receipt):
    """Process an individual receipt or document to add USD conversion"""
    if not isinstance(receipt, dict):
        return receipt
    # Create a copy to avoid modifying the original
    processed = receipt.copy()
    
    # Determine currency from existing field or infer it
    currency = processed.get('currency')
    if not currency:
        # Try to infer currency from values
        for field in ['total', 'totalPaid', 'totalDue', 'Total']:
            if field in processed and isinstance(processed[field], str):
                if '£' in processed[field]:
                    processed['currency'] = 'GBP'
                    currency = 'GBP'
                    break
                elif '€' in processed[field]:
                    processed['currency'] = 'EUR'
                    currency = 'EUR'
                    break
                elif '$' in processed[field]:
                    processed['currency'] = 'USD'
                    currency = 'USD'
                    break
    
    # Process total amount fields
    for field in ['total', 'totalPaid', 'totalDue', 'Total']:
        if field in processed:
            amount = extract_float(processed[field])
            if amount is not None:
                # Update the field with clean numeric value
                processed[field] = amount
                
                # Add USD conversion if not already USD
                if currency and currency != 'USD':
                    processed[f"{field}_USD"] = cur_converter(amount, currency)
                    # Add TOTAL_USD field for consistency
                    if field.lower() in ['total', 'totalpaid', 'totaldue']:
                        processed['TOTAL_USD'] = cur_converter(amount, currency)
                else:
                    processed['TOTAL_USD'] = amount
    # Process line items
    if 'items' in processed and isinstance(processed['items'], list):
        for item in processed['items']:
            if isinstance(item, dict):
                for amount_field in ['amount', 'price']:
                    if amount_field in item:
                        amount = extract_float(item[amount_field])
                        if amount is not None:
                            item[amount_field] = amount
                            # Convert item amount if receipt has currency
                            if currency and currency != 'USD':
                                item[f"{amount_field}_USD"] = cur_converter(amount, currency)
    
    return processed

