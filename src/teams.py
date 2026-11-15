"""Intent -> team routing table (grouped by CLINC domain; appliance intents get their own teams)."""
from config import HUMAN_TEAM, OOS_INTENT

_GROUPS = {
    "Banking & Cards": """account_blocked redeem_rewards interest_rate report_lost_card freeze_account
        bill_balance improve_credit_score credit_limit balance exchange_rate card_declined rewards_balance
        bill_due transactions credit_score report_fraud spending_history damaged_card pin_change
        replacement_card_duration new_card credit_limit_change pay_bill order_checks apr transfer
        min_payment international_fees routing international_visa""",
    "HR & Workplace": """schedule_meeting meeting_schedule pto_request pto_request_status pto_balance pto_used
        application_status payday income taxes w2 direct_deposit rollover_401k insurance insurance_change
        calendar calendar_update todo_list todo_list_update reminder reminder_update next_holiday""",
    "Travel & Transport": """oil_change_how oil_change_when gas_type gas jump_start tire_change tire_pressure
        traffic mpg directions distance last_maintenance schedule_maintenance uber car_rental book_flight
        flight_status carry_on lost_luggage book_hotel plug_type timezone travel_alert travel_notification
        travel_suggestion vaccines share_location current_location""",
    "Food & Dining": """restaurant_reviews nutrition_info accept_reservations restaurant_reservation
        cancel_reservation confirm_reservation restaurant_suggestion meal_suggestion food_last cook_time
        recipe ingredients_list ingredient_substitution calories expiration_date how_busy order order_status""",
    "Smart Home & Assistant": """weather smart_home text make_call timer flip_coin roll_dice calculator
        measurement_conversion definition spelling translate play_music next_song update_playlist what_song
        change_volume change_speed whisper_mode change_language change_accent change_ai_name change_user_name
        user_name repeat sync_device find_phone reset_settings alarm date time shopping_list
        shopping_list_update what_can_i_ask_you cancel""",
    "Auto reply": """greeting goodbye thank_you yes no maybe are_you_a_bot what_are_your_hobbies
        meaning_of_life do_you_have_pets tell_joke fun_fact how_old_are_you what_is_your_name who_made_you
        who_do_you_work_for where_are_you_from""",
    "Technical Service": "not_draining not_heating error_code",
    "Spare Parts": "spare_part_order",
    "Warranty": "warranty_claim",
    HUMAN_TEAM: OOS_INTENT,
}

TEAM_OF = {intent: team for team, names in _GROUPS.items() for intent in names.split()}
