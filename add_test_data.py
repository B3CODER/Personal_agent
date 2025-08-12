#!/usr/bin/env python3
"""
Add test data to MongoDB for testing
"""
import os
from decimal import Decimal
from datetime import datetime, timezone
from dotenv import load_dotenv
from main import FinanceProfile, upsert_user_profile

load_dotenv()

def add_test_data():
    """Add multiple test profiles to MongoDB"""
    
    test_profiles = [
        {
            "user_id": "alice_456",
            "profile": FinanceProfile(
                user_age_years=25,
                user_occupation_job="Marketing Manager",
                user_location_city_country="New York, NY",
                income_total_yearly_amount=Decimal("85000.00"),
                income_monthly_take_home=Decimal("6000.00"),
                expense_total_monthly=Decimal("3500.00"),
                expense_housing_rent_mortgage=Decimal("2000.00"),
                expense_food_groceries_dining=Decimal("600.00"),
                savings_monthly_amount=Decimal("1500.00"),
                savings_emergency_fund_total=Decimal("8000.00"),
                investment_total_portfolio_value=Decimal("25000.00"),
                investment_crypto_value=Decimal("2000.00"),
                debt_credit_card_balance=Decimal("1500.00"),
                prefers_simple_explanations=True,
                prefers_detailed_analysis=False,
                profile_last_updated_date=datetime.now(timezone.utc)
            )
        },
        {
            "user_id": "bob_789",
            "profile": FinanceProfile(
                user_age_years=35,
                user_occupation_job="Financial Analyst",
                user_location_city_country="Chicago, IL",
                income_total_yearly_amount=Decimal("110000.00"),
                income_monthly_take_home=Decimal("7500.00"),
                expense_total_monthly=Decimal("4500.00"),
                expense_housing_rent_mortgage=Decimal("2200.00"),
                expense_food_groceries_dining=Decimal("700.00"),
                savings_monthly_amount=Decimal("2500.00"),
                savings_emergency_fund_total=Decimal("20000.00"),
                investment_total_portfolio_value=Decimal("75000.00"),
                investment_crypto_value=Decimal("8000.00"),
                debt_credit_card_balance=Decimal("0.00"),
                prefers_simple_explanations=False,
                prefers_detailed_analysis=True,
                profile_last_updated_date=datetime.now(timezone.utc)
            )
        },
        {
            "user_id": "charlie_101",
            "profile": FinanceProfile(
                user_age_years=22,
                user_occupation_job="Graduate Student",
                user_location_city_country="Austin, TX",
                income_total_yearly_amount=Decimal("30000.00"),
                income_monthly_take_home=Decimal("2200.00"),
                expense_total_monthly=Decimal("1800.00"),
                expense_housing_rent_mortgage=Decimal("800.00"),
                expense_food_groceries_dining=Decimal("400.00"),
                savings_monthly_amount=Decimal("200.00"),
                savings_emergency_fund_total=Decimal("1000.00"),
                investment_total_portfolio_value=Decimal("0.00"),
                investment_crypto_value=Decimal("500.00"),
                debt_credit_card_balance=Decimal("3000.00"),
                prefers_simple_explanations=True,
                prefers_detailed_analysis=False,
                profile_last_updated_date=datetime.now(timezone.utc)
            )
        }
    ]
    
    print("Adding test profiles to MongoDB...")
    
    for test_data in test_profiles:
        user_id = test_data["user_id"]
        profile = test_data["profile"]
        
        try:
            profile_dict = profile.model_dump(mode='json')
            upsert_user_profile(user_id, profile_dict)
            print(f"✅ Added profile for {user_id}")
        except Exception as e:
            print(f"❌ Error adding profile for {user_id}: {e}")
    
    print("\n🎉 Test data added successfully!")
    print("Check MongoDB Compass to see the new documents in the 'finance_profiles' collection.")

if __name__ == "__main__":
    add_test_data() 