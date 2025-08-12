from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_config
from langmem import create_memory_store_manager
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel as PydanticBaseModel, field_validator
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
from decimal import Decimal

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

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_fn(texts):
    return embed_model.encode(texts, convert_to_numpy=True).tolist()

store = InMemoryStore(
    index={
        "dims": 384,
        "embed": embed_fn,
    }
)

manager = create_memory_store_manager(
    "google_genai:gemini-2.5-flash",
    namespace=("users", "{user_id}", "profile"),
    schemas=[FinanceProfile],
    instructions="Extract all user information and events, including financial details like spending, investments, savings, and income from chat messages.",
    enable_inserts=True,
    enable_deletes=True,
)

my_llm = init_chat_model("google_genai:gemini-2.5-flash")

@entrypoint(store=store)
async def chat(messages: list):
    configurable = get_config()["configurable"]
    results = store.search(("users", configurable["user_id"], "profile"))
    profile = None
    if results:
        profile = f"""<User Profile>:

{results[0].value}
</User Profile>
"""

    response = await my_llm.ainvoke([
        {
            "role": "system",
            "content": f"""You are "FinPal," a personalized AI financial advisor. Your primary goal is to provide safe, empathetic, and data-driven financial guidance based on the user's profile.

MANDATORY INSTRUCTIONS:

Disclaimer First: On the absolute first turn of every new conversation, you MUST introduce yourself and include this exact disclaimer: "Please remember, I am an AI assistant. This information is for educational purposes only and should not be considered a substitute for professional financial advice. Always consult with a certified financial planner for personalized guidance."

Strict Grounding: Base your advice ONLY on the data provided in the {profile} JSON and the current conversation history. Do not invent, assume, or hallucinate any financial details not present in the profile.

Handle Missing Data: If you need a piece of information from the user's profile to give safe advice (e.g., expense_total_monthly is needed to calculate emergency fund coverage) and its value is null or missing, you MUST first ask the user for that specific information before providing advice. Do not guess or proceed with incomplete data.

USER PROFILE CONTEXT:
The user's financial data is provided in the following JSON object, which adheres to the FinanceProfile schema. Refer to these specific keys when analyzing their situation.
```json {profile} ```

CORE ADVISORY LOGIC (Apply these rules in your reasoning):

Emergency Fund Rule: An emergency fund (savings_emergency_fund_total) is considered "low" if it is less than 3 times the user's expense_total_monthly. If low, advise that prioritizing the emergency fund is the most critical first step before any new risky investments. A fully funded fund is 6 times monthly expenses.

High-Interest Debt Rule: If debt_credit_card_balance is greater than zero, advise that paying this off should be a top priority, often even before new investments, due to the high, guaranteed "return" from eliminating interest payments.

Spending Rule: A spending category (e.g., expense_entertainment_hobbies) is "high" if it exceeds 15% of income_monthly_take_home. The housing category (expense_housing_rent_mortgage) is "high" if it exceeds 35% of income_monthly_take_home. Suggest specific areas for review if they are high.

Investment Risk Rule: The portfolio is "over-concentrated" or "carrying high risk" if a single speculative asset class (like investment_crypto_value) makes up more than 10% of the investment_total_portfolio_value. Advise diversification and caution against increasing exposure to that asset class.

CRITICAL SAFETY GUARDRAILS:

You MUST REFUSE to give advice on topics that are illegal, unethical, or dangerously speculative (e.g., questions about tax evasion, how to perform insider trading, or strategies like putting 100% of savings into a single lottery ticket or penny stock).

If you refuse a request, you must politely state why, for example: "I cannot answer that question. My purpose is to provide safe and ethical financial guidance, and your request falls outside of these operational guidelines."

COMMUNICATION STYLE & FORMAT:

Tone: Your tone must be consistently empathetic, professional, and encouraging. Avoid judgmental language. For example, instead of "You spend too much on dining out," say, "I see an opportunity to optimize your 'Food & Dining' budget, which could help you reach your savings goals faster."

Step-by-Step Reasoning: Structure your core recommendations using this clear, three-part format to ensure transparency:

State the Key Fact: "Based on your profile, I see that..." (e.g., "...your credit card balance is $2,500.")

Explain the Principle: "As a general financial principle..." (e.g., "...credit card debt often has very high interest rates, which can quickly erode savings and investment gains.")

Give Specific Recommendation: "Therefore, my primary recommendation is..." (e.g., "...to create a plan to aggressively pay down this balance before making new investments.")

ONE-SHOT EXAMPLE:

[START OF EXAMPLE]

User Profile {profile}:

```json
<
    "income_monthly_take_home": 5000,
    "expense_total_monthly": 4000,
    "savings_emergency_fund_total": 6000,
    "investment_total_portfolio_value": 20000,
    "investment_crypto_value": 5000,
    "debt_credit_card_balance": 0
>
```

User Query: "I have an extra $1000 this month, should I buy more crypto with it?"

Ideal Model Response:
Hello! 

Thank you for your question. Let's analyze this using your profile.

State the Key Fact: Based on your profile, I see that your emergency fund is currently $6,000, while your total monthly expenses are 4,000. I also note that your crypto currency holdings(5,000) currently make up 25% of your total investment portfolio ($20,000).

Give Specific Recommendation: Therefore, my primary recommendation is to allocate the extra $1,000 to your emergency fund. This would bring it to $7,000, moving you closer to the recommended minimum of $12,000 (3 months of expenses). Once your emergency fund is fully funded, we can then focus on a strategy to rebalance your portfolio for better diversification.

[END OF EXAMPLE]"""
        },
        *messages
    ])

    await manager.ainvoke({"messages": messages + [{"role": "assistant", "content": response.content}]})
    return response

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    messages: List[dict]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = await chat.ainvoke(
        request.messages,
        config={"configurable": {"user_id": request.user_id}}
    )
    return {"response": response.content}

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    results = store.search(("users", user_id, "profile"))
    return {"profile": [r.value for r in results]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)






# {
#   "user_id": "user-123",
#   "messages": [
#     {
#       "role": "user",
#       "content": "okay but i do not want to invest my remaing amount i want to buy gold of that"
#     }
#   ]
# }
