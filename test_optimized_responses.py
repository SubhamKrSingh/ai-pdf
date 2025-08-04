#!/usr/bin/env python3
"""
Test script to demonstrate optimized response generation.
This script shows how the optimized LLM service generates more concise answers.
"""

import asyncio
import json
from typing import List

from app.models.schemas import DocumentChunk
from app.services.llm_service import get_llm_service


async def test_optimized_responses():
    """Test the optimized response generation with sample data."""
    
    # Sample context chunks (simulating policy document content)
    sample_chunks = [
        DocumentChunk(
            id="chunk_1",
            document_id="policy_doc",
            content="The Grace Period for payment of the premium shall be thirty days. This grace period allows for the renewal or continuation of the policy without losing continuity benefits related to waiting periods and coverage of pre-existing diseases. However, coverage is not available during the period for which no premium is received.",
            metadata={"page_number": 4, "section": "Grace Period"},
            chunk_index=1
        ),
        DocumentChunk(
            id="chunk_2", 
            document_id="policy_doc",
            content="The waiting period for pre-existing diseases (PED) to be covered is 36 months of continuous coverage after the date of inception of the first policy with the company. Coverage under the policy after this waiting period is subject to the PED being declared at the time of application and accepted by the company.",
            metadata={"page_number": 5, "section": "Waiting Periods"},
            chunk_index=2
        )
    ]
    
    # Sample questions
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]
    
    # Expected concise answers
    expected_answers = [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
    ]
    
    print("Testing Optimized Response Generation")
    print("=" * 50)
    
    try:
        llm_service = get_llm_service()
        
        for i, question in enumerate(questions):
            print(f"\nQuestion {i+1}: {question}")
            print("-" * 40)
            
            # Generate answer using optimized service
            answer = await llm_service.generate_contextual_answer(
                question=question,
                context_chunks=sample_chunks
            )
            
            print(f"Generated Answer: {answer}")
            print(f"Expected Answer: {expected_answers[i]}")
            print(f"Length - Generated: {len(answer)}, Expected: {len(expected_answers[i])}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")


def compare_response_formats():
    """Compare old vs new response formats."""
    
    sample_request = {
        "documents": "https://example.com/policy.pdf",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    # Old verbose format (your current output)
    old_format = {
        "answers": [
            "The grace period for premium payment under the National Parivar Mediclaim Plus Policy is thirty days, as stated in section 2.21: \"The Grace Period for payment of the premium shall be thirty days\" (Context Chunk 4). This grace period allows for the renewal or continuation of the policy without losing continuity benefits related to waiting periods and coverage of pre-existing diseases (Context Chunk 4). However, coverage is not available during the period for which no premium is received (Context Chunk 4).",
            "The waiting period for pre-existing diseases (PED) to be covered is 36 months of continuous coverage after the date of inception of the first policy with the company (4.1.a). Coverage under the policy after this waiting period is subject to the PED being declared at the time of application and accepted by the company (4.1.d)."
        ]
    }
    
    # New optimized format (expected output)
    new_format = {
        "answers": [
            "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
        ]
    }
    
    print("\nResponse Format Comparison")
    print("=" * 50)
    
    print("\nOLD FORMAT (Verbose):")
    print(json.dumps(old_format, indent=2))
    
    print(f"\nOLD FORMAT STATS:")
    print(f"Answer 1 length: {len(old_format['answers'][0])} characters")
    print(f"Answer 2 length: {len(old_format['answers'][1])} characters")
    print(f"Total length: {sum(len(a) for a in old_format['answers'])} characters")
    
    print("\nNEW FORMAT (Concise):")
    print(json.dumps(new_format, indent=2))
    
    print(f"\nNEW FORMAT STATS:")
    print(f"Answer 1 length: {len(new_format['answers'][0])} characters")
    print(f"Answer 2 length: {len(new_format['answers'][1])} characters")
    print(f"Total length: {sum(len(a) for a in new_format['answers'])} characters")
    
    # Calculate improvement
    old_total = sum(len(a) for a in old_format['answers'])
    new_total = sum(len(a) for a in new_format['answers'])
    improvement = ((old_total - new_total) / old_total) * 100
    
    print(f"\nIMPROVEMENT:")
    print(f"Reduction: {improvement:.1f}% shorter responses")
    print(f"Saved: {old_total - new_total} characters")


if __name__ == "__main__":
    print("LLM Response Optimization Test")
    print("=" * 50)
    
    # Show format comparison
    compare_response_formats()
    
    # Test actual generation (requires API key)
    print("\n" + "=" * 50)
    print("To test actual generation, run:")
    print("python -c \"import asyncio; from test_optimized_responses import test_optimized_responses; asyncio.run(test_optimized_responses())\"")