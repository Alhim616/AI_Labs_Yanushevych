data = {
    "Outlook": {
        "rainy": {"Yes": 2, "No": 3},
        "sunny": {"Yes": 3, "No": 2},
        "overcast": {"Yes": 4, "No": 0}
    },
    "Humidity": {
        "high": {"Yes": 3, "No": 4},
        "normal": {"Yes": 6, "No": 1}
    },
    "Wind": {
        "weak": {"Yes": 6, "No": 2},
        "strong": {"Yes": 3, "No": 3}
    }
}

total_yes = 9
total_no = 5
total = total_yes + total_no


p_yes = total_yes / total
p_no = total_no / total

conditions = {
    "Outlook": input("Enter outlook (sunny, overcast, rainy): ").lower(),
    "Humidity": input("Enter humidity level (high, normal): ").lower(),
    "Wind": input("Enter wind strength (weak, strong): ").lower()
}

p_rain_yes = data["Outlook"][conditions["Outlook"]]["Yes"] / total_yes
p_rain_no = data["Outlook"][conditions["Outlook"]]["No"] / total_no

p_humidity_yes = data["Humidity"][conditions["Humidity"]]["Yes"] / total_yes
p_humidity_no = data["Humidity"][conditions["Humidity"]]["No"] / total_no

p_wind_yes = data["Wind"][conditions["Wind"]]["Yes"] / total_yes
p_wind_no = data["Wind"][conditions["Wind"]]["No"] / total_no

p_yes_given_conditions = p_rain_yes * p_humidity_yes * p_wind_yes * p_yes
p_no_given_conditions = p_rain_no * p_humidity_no * p_wind_no * p_no


total_probability = p_yes_given_conditions + p_no_given_conditions
p_yes_final = p_yes_given_conditions / total_probability
p_no_final = p_no_given_conditions / total_probability

print(f"Ймовірність, що матч відбудеться (Yes): {p_yes_final:.2f}")
print(f"Ймовірність, що матч не відбудеться (No): {p_no_final:.2f}")
