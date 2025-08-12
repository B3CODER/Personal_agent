import os
import asyncio
from typing import List, Optional
from decimal import Decimal
from datetime import datetime, timezone
from pydantic import BaseModel, field_serializer
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from pymongo import MongoClient

from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint
from langgraph.config import get_config
from langmem import create_memory_store_manager

load_dotenv()

# ----------------------
# Config
# ----------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "finpal_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finance_profiles")
MEMORY_LLM = os.getenv("MEMORY_LLM", "google_genai:gemini-2.5-flash")  # use same model for everything

# ----------------------
# FinanceProfile schema
# ----------------------
class FinanceProfile(BaseModel):    
    # Basic Info - "Tell me about yourself"
    user_age_years: Optional[int] = None
    user_occupation_job: Optional[str] = None
    user_location_city_country: Optional[str] = None
    user_has_spouse_partner: Optional[bool] = None
    user_number_of_children: Optional[int] = None
    
    # Income - "How much do you earn?"
    income_total_yearly_amount: Optional[Decimal] = None
    income_monthly_take_home: Optional[Decimal] = None
    income_from_salary: Optional[Decimal] = None
    income_from_business: Optional[Decimal] = None
    income_from_investments: Optional[Decimal] = None
    income_from_rental: Optional[Decimal] = None
    income_other_sources: Optional[str] = None
    
    # Expenses - "What are your monthly expenses?"
    expense_total_monthly: Optional[Decimal] = None
    expense_housing_rent_mortgage: Optional[Decimal] = None
    expense_food_groceries_dining: Optional[Decimal] = None
    expense_transportation_car_gas: Optional[Decimal] = None
    expense_utilities_bills: Optional[Decimal] = None
    expense_entertainment_hobbies: Optional[Decimal] = None
    expense_other_categories: Optional[str] = None
    
    # Savings - "How much do you save?"
    savings_monthly_amount: Optional[Decimal] = None
    savings_emergency_fund_total: Optional[Decimal] = None
    savings_percentage_of_income: Optional[float] = None
    
    # Goals - "What are your financial goals?"
    goal_buy_house_amount_needed: Optional[Decimal] = None
    goal_buy_house_timeline_years: Optional[int] = None
    goal_retirement_target_amount: Optional[Decimal] = None
    goal_retirement_age_target: Optional[int] = None
    goal_children_education_amount: Optional[Decimal] = None
    goal_vacation_travel_budget: Optional[Decimal] = None
    goal_other_major_purchases: Optional[str] = None
    goal_most_important_priority: Optional[str] = None
    
    # Current Investments - "What investments do you have?"
    investment_total_portfolio_value: Optional[Decimal] = None
    investment_stocks_value: Optional[Decimal] = None
    investment_bonds_value: Optional[Decimal] = None
    investment_mutual_funds_value: Optional[Decimal] = None
    investment_etf_value: Optional[Decimal] = None
    investment_crypto_value: Optional[Decimal] = None
    investment_real_estate_value: Optional[Decimal] = None
    investment_gold_precious_metals: Optional[Decimal] = None
    investment_fixed_deposits_value: Optional[Decimal] = None
    investment_retirement_401k_ira: Optional[Decimal] = None
    
    # Debt - "What debts do you have?"
    debt_total_amount_owed: Optional[Decimal] = None
    debt_home_loan_mortgage: Optional[Decimal] = None
    debt_car_loan_amount: Optional[Decimal] = None
    debt_student_loan_amount: Optional[Decimal] = None
    debt_credit_card_balance: Optional[Decimal] = None
    debt_personal_loan_amount: Optional[Decimal] = None
    debt_monthly_total_payments: Optional[Decimal] = None
    
    # Risk Profile - "What's your risk tolerance?"
    risk_tolerance_level: Optional[str] = None  # "low", "medium", "high"
    risk_can_afford_to_lose_amount: Optional[Decimal] = None
    risk_investment_experience_years: Optional[int] = None
    risk_reaction_to_market_drop: Optional[str] = None  # "sell", "hold", "buy more"
    
    # Investment Preferences - "How do you like to invest?"
    prefer_stocks_over_bonds: Optional[bool] = None
    prefer_domestic_over_international: Optional[bool] = None
    prefer_growth_over_dividend: Optional[bool] = None
    prefer_individual_stocks_over_funds: Optional[bool] = None
    prefer_active_over_passive: Optional[bool] = None
    interested_in_crypto: Optional[bool] = None
    interested_in_real_estate: Optional[bool] = None
    avoid_specific_sectors: Optional[List[str]] = None
    
    # Tax Situation - "What's your tax situation?"
    tax_bracket_percentage: Optional[float] = None
    tax_filing_status: Optional[str] = None  # "single", "married", etc.
    tax_state_residence: Optional[str] = None
    tax_deductions_claimed: Optional[List[str]] = None
    
    # Insurance - "What insurance do you have?"
    insurance_has_life: Optional[bool] = None
    insurance_has_health: Optional[bool] = None
    insurance_has_disability: Optional[bool] = None
    insurance_has_home_renters: Optional[bool] = None
    insurance_has_auto: Optional[bool] = None
    
    # Financial Knowledge - "How much do you know about investing?"
    knowledge_understands_stocks: Optional[bool] = None
    knowledge_understands_bonds: Optional[bool] = None
    knowledge_understands_mutual_funds: Optional[bool] = None
    knowledge_needs_basic_education: Optional[bool] = None
    knowledge_confident_level: Optional[str] = None  # "beginner", "intermediate", "advanced"
    
    # Past Behavior - "Tell me about your investment history"
    history_best_investment_made: Optional[str] = None
    history_worst_investment_made: Optional[str] = None
    history_learned_lessons: Optional[str] = None
    history_years_investing: Optional[int] = None
    
    # Future Plans - "What are your future plans?"
    plans_major_purchase_next_year: Optional[str] = None
    plans_career_change_expected: Optional[bool] = None
    plans_expecting_inheritance: Optional[bool] = None
    plans_starting_business: Optional[bool] = None
    
    # Preferences - "How do you want advice?"
    prefers_simple_explanations: Optional[bool] = None
    prefers_detailed_analysis: Optional[bool] = None
    prefers_conservative_advice: Optional[bool] = None
    prefers_aggressive_strategies: Optional[bool] = None
    
    # Tracking - "When did we last update?"
    profile_last_updated_date: Optional[datetime] = None
    portfolio_last_reviewed_date: Optional[datetime] = None
    goals_last_discussed_date: Optional[datetime] = None
    
    # Open Notes - "Anything else to remember?"
    notes_special_circumstances: Optional[str] = None
    notes_specific_questions_asked: Optional[List[str]] = None
    notes_advice_given_previously: Optional[List[str]] = None

    # Field serializers for all Decimal fields
    @field_serializer('income_total_yearly_amount')
    def serialize_income_total_yearly_amount(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('income_monthly_take_home')
    def serialize_income_monthly_take_home(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('income_from_salary')
    def serialize_income_from_salary(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('income_from_business')
    def serialize_income_from_business(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('income_from_investments')
    def serialize_income_from_investments(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('income_from_rental')
    def serialize_income_from_rental(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('expense_total_monthly')
    def serialize_expense_total_monthly(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('expense_housing_rent_mortgage')
    def serialize_expense_housing_rent_mortgage(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('expense_food_groceries_dining')
    def serialize_expense_food_groceries_dining(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('expense_transportation_car_gas')
    def serialize_expense_transportation_car_gas(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('expense_utilities_bills')
    def serialize_expense_utilities_bills(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('expense_entertainment_hobbies')
    def serialize_expense_entertainment_hobbies(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('savings_monthly_amount')
    def serialize_savings_monthly_amount(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('savings_emergency_fund_total')
    def serialize_savings_emergency_fund_total(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('goal_buy_house_amount_needed')
    def serialize_goal_buy_house_amount_needed(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('goal_retirement_target_amount')
    def serialize_goal_retirement_target_amount(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('goal_children_education_amount')
    def serialize_goal_children_education_amount(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('goal_vacation_travel_budget')
    def serialize_goal_vacation_travel_budget(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_total_portfolio_value')
    def serialize_investment_total_portfolio_value(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_stocks_value')
    def serialize_investment_stocks_value(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_bonds_value')
    def serialize_investment_bonds_value(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_mutual_funds_value')
    def serialize_investment_mutual_funds_value(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_etf_value')
    def serialize_investment_etf_value(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_crypto_value')
    def serialize_investment_crypto_value(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_real_estate_value')
    def serialize_investment_real_estate_value(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_gold_precious_metals')
    def serialize_investment_gold_precious_metals(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_fixed_deposits_value')
    def serialize_investment_fixed_deposits_value(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('investment_retirement_401k_ira')
    def serialize_investment_retirement_401k_ira(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('debt_total_amount_owed')
    def serialize_debt_total_amount_owed(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('debt_home_loan_mortgage')
    def serialize_debt_home_loan_mortgage(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('debt_car_loan_amount')
    def serialize_debt_car_loan_amount(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('debt_student_loan_amount')
    def serialize_debt_student_loan_amount(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('debt_credit_card_balance')
    def serialize_debt_credit_card_balance(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('debt_personal_loan_amount')
    def serialize_debt_personal_loan_amount(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('debt_monthly_total_payments')
    def serialize_debt_monthly_total_payments(self, value):
        return float(value) if value is not None else None
    
    @field_serializer('risk_can_afford_to_lose_amount')
    def serialize_risk_can_afford_to_lose_amount(self, value):
        return float(value) if value is not None else None

# ----------------------
# MongoDB helper
# ----------------------
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def get_user_profile(user_id: str) -> Optional[dict]:
    return collection.find_one({"user_id": user_id}, {"_id": 0})

def upsert_user_profile(user_id: str, profile_data: dict):
    profile_data["user_id"] = user_id
    profile_data["profile_last_updated_date"] = datetime.now(timezone.utc)
    collection.update_one({"user_id": user_id}, {"$set": profile_data}, upsert=True)

# ----------------------
# Embedding (still needed for LangMem if you want vector stuff later)
# ----------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed_fn(texts: List[str]):
    return embed_model.encode(texts, convert_to_numpy=True).tolist()

# ----------------------
# Memory Manager (extract profile info automatically)
# ----------------------
manager = create_memory_store_manager(
    MEMORY_LLM,
    namespace=("users", "{user_id}", "profile"),
    schemas=[FinanceProfile],
    instructions="Extract and update FinanceProfile info from the conversation.",
    enable_inserts=True,
    enable_deletes=True,
)

# ----------------------
# LLM + Checkpointer
# ----------------------
my_graph_llm = init_chat_model(model=MEMORY_LLM)
checkpointer = MemorySaver()  # in-memory checkpoints

# ----------------------
# LangGraph node
# ----------------------
async def call_model(state: MessagesState):
    user_id = get_config().get("configurable", {}).get("user_id", "default_user")

    profile = get_user_profile(user_id)
    if profile:
        system_prompt = f"You are FinPal. Use this profile:\n{profile}"
    else:
        system_prompt = "You are FinPal, a helpful financial assistant."

    messages_for_model = [{"role": "system", "content": system_prompt}] + state["messages"]

    response = my_graph_llm.invoke(messages_for_model)
    if hasattr(response, "content"):
        assistant_message = {"role": "assistant", "content": response.content}
    else:
        assistant_message = {"role": "assistant", "content": str(response)}

    new_messages = state["messages"] + [assistant_message]

    # Extract profile data from conversation using LLM
    try:
        # Create a comprehensive extraction prompt
        extraction_prompt = f"""
        Extract financial profile information from this conversation and return it as a JSON object.
        Only extract information that is explicitly mentioned or can be reasonably inferred.
        
        Conversation:
        {new_messages}
        
        Return a JSON object with these fields (only if mentioned or inferred):
        
        # Basic Info
        - user_age_years: age in years
        - user_occupation_job: job title/occupation
        - user_location_city_country: location
        - user_has_spouse_partner: true/false if mentioned
        - user_number_of_children: number of children if mentioned
        
        # Income
        - income_total_yearly_amount: yearly income as number
        - income_monthly_take_home: monthly take-home pay as number
        - income_from_salary: salary income as number
        - income_from_business: business income as number
        - income_from_investments: investment income as number
        - income_from_rental: rental income as number
        - income_other_sources: other income sources as string
        
        # Expenses
        - expense_total_monthly: monthly expenses as number
        - expense_housing_rent_mortgage: housing costs as number
        - expense_food_groceries_dining: food costs as number
        - expense_transportation_car_gas: transportation costs as number
        - expense_utilities_bills: utility bills as number
        - expense_entertainment_hobbies: entertainment costs as number
        - expense_other_categories: other expense categories as string
        
        # Savings
        - savings_monthly_amount: monthly savings as number
        - savings_emergency_fund_total: emergency fund as number
        - savings_percentage_of_income: savings percentage as float
        
        # Goals
        - goal_buy_house_amount_needed: house purchase amount as number
        - goal_buy_house_timeline_years: house purchase timeline as number
        - goal_retirement_target_amount: retirement target as number
        - goal_retirement_age_target: retirement age as number
        - goal_children_education_amount: education fund as number
        - goal_vacation_travel_budget: travel budget as number
        - goal_other_major_purchases: other major purchases as string
        - goal_most_important_priority: most important goal as string
        
        # Investments
        - investment_total_portfolio_value: total portfolio value as number
        - investment_stocks_value: stocks value as number
        - investment_bonds_value: bonds value as number
        - investment_mutual_funds_value: mutual funds value as number
        - investment_etf_value: ETF value as number
        - investment_crypto_value: crypto value as number
        - investment_real_estate_value: real estate value as number
        - investment_gold_precious_metals: precious metals value as number
        - investment_fixed_deposits_value: fixed deposits value as number
        - investment_retirement_401k_ira: retirement accounts value as number
        
        # Debt
        - debt_total_amount_owed: total debt as number
        - debt_home_loan_mortgage: mortgage amount as number
        - debt_car_loan_amount: car loan amount as number
        - debt_student_loan_amount: student loan amount as number
        - debt_credit_card_balance: credit card balance as number
        - debt_personal_loan_amount: personal loan amount as number
        - debt_monthly_total_payments: monthly debt payments as number
        
        # Risk Profile
        - risk_tolerance_level: "low", "medium", or "high"
        - risk_can_afford_to_lose_amount: amount can afford to lose as number
        - risk_investment_experience_years: years of investment experience as number
        - risk_reaction_to_market_drop: "sell", "hold", or "buy more"
        
        # Investment Preferences
        - prefer_stocks_over_bonds: true/false
        - prefer_domestic_over_international: true/false
        - prefer_growth_over_dividend: true/false
        - prefer_individual_stocks_over_funds: true/false
        - prefer_active_over_passive: true/false
        - interested_in_crypto: true/false
        - interested_in_real_estate: true/false
        - avoid_specific_sectors: array of sector names
        
        # Tax Situation
        - tax_bracket_percentage: tax bracket as float
        - tax_filing_status: "single", "married", etc.
        - tax_state_residence: state of residence
        - tax_deductions_claimed: array of deduction names
        
        # Insurance
        - insurance_has_life: true/false
        - insurance_has_health: true/false
        - insurance_has_disability: true/false
        - insurance_has_home_renters: true/false
        - insurance_has_auto: true/false
        
        # Financial Knowledge
        - knowledge_understands_stocks: true/false
        - knowledge_understands_bonds: true/false
        - knowledge_understands_mutual_funds: true/false
        - knowledge_needs_basic_education: true/false
        - knowledge_confident_level: "beginner", "intermediate", or "advanced"
        
        # Past Behavior
        - history_best_investment_made: description of best investment
        - history_worst_investment_made: description of worst investment
        - history_learned_lessons: lessons learned as string
        - history_years_investing: years of investing as number
        
        # Future Plans
        - plans_major_purchase_next_year: major purchase plans as string
        - plans_career_change_expected: true/false
        - plans_expecting_inheritance: true/false
        - plans_starting_business: true/false
        
        # Preferences
        - prefers_simple_explanations: true/false
        - prefers_detailed_analysis: true/false
        - prefers_conservative_advice: true/false
        - prefers_aggressive_strategies: true/false
        
        # Notes
        - notes_special_circumstances: special circumstances as string
        - notes_specific_questions_asked: array of questions asked
        - notes_advice_given_previously: array of advice given
        
        Return only the JSON object, nothing else. Only include fields that are mentioned or can be reasonably inferred from the conversation.
        """
        
        # Use the LLM to extract data
        extraction_messages = [{"role": "user", "content": extraction_prompt}]
        extraction_response = my_graph_llm.invoke(extraction_messages)
        
        if hasattr(extraction_response, 'content'):
            extraction_text = extraction_response.content
        else:
            extraction_text = str(extraction_response)
        
        print(f"Extraction response: {extraction_text}")
        
        # Try to parse JSON from the response
        import json
        import re
        
        # Find JSON in the response
        json_match = re.search(r'\{.*\}', extraction_text, re.DOTALL)
        if json_match:
            try:
                extracted_data = json.loads(json_match.group())
                print(f"✅ Successfully extracted data: {extracted_data}")
                
                # Convert any Decimal values to float for MongoDB
                for key, value in extracted_data.items():
                    if isinstance(value, (int, float)) and key.endswith('_amount') or key.endswith('_value') or key.endswith('_balance') or key.endswith('_total'):
                        extracted_data[key] = float(value)
                
                extracted_data["profile_last_updated_date"] = datetime.now(timezone.utc)
                upsert_user_profile(user_id, extracted_data)
                print(f"✅ Updated profile for user {user_id}: {extracted_data}")
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"Raw extraction text: {extraction_text}")
        else:
            print(f"⚠️ No JSON found in extraction response")
            print(f"Raw extraction text: {extraction_text}")
            
    except Exception as e:
        print(f"❌ Error extracting profile data: {e}")
        import traceback
        traceback.print_exc()

    return {"messages": new_messages}

# ----------------------
# Graph setup
# ----------------------
builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    messages: List[dict]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {
        "configurable": {
            "user_id": request.user_id,
            "thread_id": request.user_id  # or some other conversation ID
        }
    }
    result = await graph.ainvoke({"messages": request.messages}, config)
    last_msg = result["messages"][-1]
    return {"response": last_msg}


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    return {"profile": get_user_profile(user_id)}

@app.post("/update_profile/{user_id}")
def update_profile(user_id: str, profile: FinanceProfile):
    # Convert Pydantic model to dict with proper serialization
    profile_dict = profile.model_dump(mode='json')
    upsert_user_profile(user_id, profile_dict)
    return {"ok": True, "profile": get_user_profile(user_id)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
