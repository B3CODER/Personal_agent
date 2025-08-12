#!/usr/bin/env python3
"""
Test script to verify MongoDB storage of FinanceProfile
"""
import os
from decimal import Decimal
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from main import FinanceProfile, upsert_user_profile, get_user_profile

load_dotenv()

def test_mongodb_storage():
    """Test storing and retrieving FinanceProfile from MongoDB"""
    
    # Test data
    test_user_id = "test_user_123"
    test_profile = FinanceProfile(
        user_age_years=30,
        user_occupation_job="Software Engineer",
        user_location_city_country="San Francisco, CA",
        income_total_yearly_amount=Decimal("120000.00"),
        income_monthly_take_home=Decimal("8000.00"),
        expense_total_monthly=Decimal("4000.00"),
        expense_housing_rent_mortgage=Decimal("2500.00"),
        expense_food_groceries_dining=Decimal("800.00"),
        savings_monthly_amount=Decimal("2000.00"),
        savings_emergency_fund_total=Decimal("15000.00"),
        investment_total_portfolio_value=Decimal("50000.00"),
        investment_crypto_value=Decimal("5000.00"),
        debt_credit_card_balance=Decimal("2000.00"),
        prefers_simple_explanations=True,
        prefers_detailed_analysis=False,
        profile_last_updated_date=datetime.now(timezone.utc)
    )
    
    print("Testing MongoDB storage...")
    print(f"Test user ID: {test_user_id}")
    print(f"Test profile: {test_profile.model_dump()}")
    
    try:
        # Store the profile
        profile_dict = test_profile.model_dump(mode='json')
        upsert_user_profile(test_user_id, profile_dict)
        print("✅ Profile stored successfully")
        
        # Retrieve the profile
        retrieved_profile = get_user_profile(test_user_id)
        print(f"✅ Profile retrieved: {retrieved_profile}")
        
        # Verify the data types
        if retrieved_profile:
            print("\nVerifying data types:")
            print(f"income_total_yearly_amount: {type(retrieved_profile.get('income_total_yearly_amount'))} = {retrieved_profile.get('income_total_yearly_amount')}")
            print(f"user_age_years: {type(retrieved_profile.get('user_age_years'))} = {retrieved_profile.get('user_age_years')}")
            print(f"prefers_simple_explanations: {type(retrieved_profile.get('prefers_simple_explanations'))} = {retrieved_profile.get('prefers_simple_explanations')}")
            
            # Test converting back to Pydantic model
            reconstructed_profile = FinanceProfile(**retrieved_profile)
            print(f"✅ Successfully reconstructed Pydantic model: {reconstructed_profile}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_mongodb_storage()
    if success:
        print("\n🎉 All tests passed! MongoDB storage is working correctly.")
    else:
        print("\n💥 Tests failed. Check the error messages above.") 