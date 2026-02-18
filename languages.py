# languages.py

# --- GTSRB CLASS MAPPING (0 to 42) ---
# This dictionary maps the ID number to the Text
SIGN_TRANSLATIONS = {
    0: {'en': 'Speed Limit 20', 'hi': 'Gati 20', 'kn': 'Vega 20', 'ta': 'Vegam 20', 'te': 'Vegam 20'},
    1: {'en': 'Speed Limit 30', 'hi': 'Gati 30', 'kn': 'Vega 30', 'ta': 'Vegam 30', 'te': 'Vegam 30'},
    2: {'en': 'Speed Limit 50', 'hi': 'Gati 50', 'kn': 'Vega 50', 'ta': 'Vegam 50', 'te': 'Vegam 50'},
    3: {'en': 'Speed Limit 60', 'hi': 'Gati 60', 'kn': 'Vega 60', 'ta': 'Vegam 60', 'te': 'Vegam 60'},
    4: {'en': 'Speed Limit 70', 'hi': 'Gati 70', 'kn': 'Vega 70', 'ta': 'Vegam 70', 'te': 'Vegam 70'},
    5: {'en': 'Speed Limit 80', 'hi': 'Gati 80', 'kn': 'Vega 80', 'ta': 'Vegam 80', 'te': 'Vegam 80'},
    6: {'en': 'End of Limit 80', 'hi': 'Gati 80 Samapt', 'kn': 'Vega 80 Mugiyitu', 'ta': 'Vegam 80 Mudivurugiradhu', 'te': 'Vegam 80 Aipoyindi'},
    7: {'en': 'Speed Limit 100', 'hi': 'Gati 100', 'kn': 'Vega 100', 'ta': 'Vegam 100', 'te': 'Vegam 100'},
    8: {'en': 'Speed Limit 120', 'hi': 'Gati 120', 'kn': 'Vega 120', 'ta': 'Vegam 120', 'te': 'Vegam 120'},
    9: {'en': 'No Passing', 'hi': 'Aage Nikalna Mana', 'kn': 'Munde Hoguvudilla', 'ta': 'Mundha Thadai', 'te': 'Munduku Velladdu'},
    10: {'en': 'No Trucks Passing', 'hi': 'Truck Passing Mana', 'kn': 'Truck Hoguvudilla', 'ta': 'Lorry Mundha Thadai', 'te': 'Lorry Velladdu'},
    11: {'en': 'Right of Way', 'hi': 'Raasta', 'kn': 'Daari Ide', 'ta': 'Vali Undu', 'te': 'Daari Undi'},
    12: {'en': 'Priority Road', 'hi': 'Prathmikta Sadak', 'kn': 'Pradhana Raste', 'ta': 'Mukkiya Saalai', 'te': 'Pradhana Road'},
    13: {'en': 'Yield', 'hi': 'Raasta Dein', 'kn': 'Daari Bidi', 'ta': 'Vali Vidu', 'te': 'Daari Ivvandi'},
    14: {'en': 'Stop', 'hi': 'Rukiye', 'kn': 'Nilli', 'ta': 'Nillungal', 'te': 'Aagandi'},
    15: {'en': 'No Vehicles', 'hi': 'Vaahan Mana', 'kn': 'Vaahana Nishedha', 'ta': 'Vaganam Thadai', 'te': 'Vahanam Nishedham'},
    16: {'en': 'No Trucks', 'hi': 'Truck Mana', 'kn': 'Truck Nishedha', 'ta': 'Lorry Thadai', 'te': 'Lorry Nishedham'},
    17: {'en': 'No Entry', 'hi': 'Pravesh Nishedh', 'kn': 'Pravesha Nishedha', 'ta': 'Nulaiva Thadai', 'te': 'Pravesham Ledu'},
    18: {'en': 'Caution', 'hi': 'Savdhaan', 'kn': 'Eccharike', 'ta': 'Eccharikkai', 'te': 'Jagratha'},
    19: {'en': 'Curve Left', 'hi': 'Baayein Mod', 'kn': 'Edakke Tiruvu', 'ta': 'Idadhu Valai', 'te': 'Edama Malupu'},
    20: {'en': 'Curve Right', 'hi': 'Daayein Mod', 'kn': 'Balakke Tiruvu', 'ta': 'Valadhu Valai', 'te': 'Kudi Malupu'},
    21: {'en': 'Double Curve', 'hi': 'Dohra Mod', 'kn': 'Jodi Tiruvu', 'ta': 'Irattai Valai', 'te': 'Rendu Malupulu'},
    22: {'en': 'Bumpy Road', 'hi': 'Kharab Sadak', 'kn': 'Gundi Raste', 'ta': 'Medupallam', 'te': 'Guntala Road'},
    23: {'en': 'Slippery Road', 'hi': 'Fisalan', 'kn': 'Jaaruva Raste', 'ta': 'Valukkum Saalai', 'te': 'Jaaru Road'},
    24: {'en': 'Narrow Road', 'hi': 'Sankri Sadak', 'kn': 'Kiridu Raste', 'ta': 'Kurugiya Saalai', 'te': 'Sannani Road'},
    25: {'en': 'Road Work', 'hi': 'Kaam Chalu', 'kn': 'Raste Kelasa', 'ta': 'Saalai Pani', 'te': 'Road Pani'},
    26: {'en': 'Traffic Signal', 'hi': 'Signal', 'kn': 'Signal', 'ta': 'Signal', 'te': 'Signal'},
    27: {'en': 'Pedestrians', 'hi': 'Paidalyatri', 'kn': 'Kaalunadige', 'ta': 'Nadai Pathai', 'te': 'Nadaka Daari'},
    28: {'en': 'Children Crossing', 'hi': 'Bache', 'kn': 'Makkalu', 'ta': 'Kulandhaigal', 'te': 'Pillalu'},
    29: {'en': 'Bicycles', 'hi': 'Cycle', 'kn': 'Cycle', 'ta': 'Cycle', 'te': 'Cycle'},
    30: {'en': 'Ice/Snow', 'hi': 'Barf', 'kn': 'Manju', 'ta': 'Pani', 'te': 'Manchu'},
    31: {'en': 'Wild Animals', 'hi': 'Janwar', 'kn': 'Praanigalu', 'ta': 'Vilangugal', 'te': 'Jantuvulu'},
    32: {'en': 'End Limits', 'hi': 'Seema Samapt', 'kn': 'Mithi Mugiyitu', 'ta': 'Ellai Mudivu', 'te': 'Haddu Aipoyindi'},
    33: {'en': 'Turn Right', 'hi': 'Daayein Mudiye', 'kn': 'Balakke Tirugi', 'ta': 'Valadhu Thirumba', 'te': 'Kudi Tirugandi'},
    34: {'en': 'Turn Left', 'hi': 'Baayein Mudiye', 'kn': 'Edakke Tirugi', 'ta': 'Idadhu Thirumba', 'te': 'Edama Tirugandi'},
    35: {'en': 'Ahead Only', 'hi': 'Seedha', 'kn': 'Munde Maatra', 'ta': 'Neraga Mattum', 'te': 'Munduku Matrame'},
    36: {'en': 'Straight or Right', 'hi': 'Seedha Ya Daayein', 'kn': 'Munde Athava Balakke', 'ta': 'Neraga Alladhu Valadhu', 'te': 'Munduku Leda Kudi'},
    37: {'en': 'Straight or Left', 'hi': 'Seedha Ya Baayein', 'kn': 'Munde Athava Edakke', 'ta': 'Neraga Alladhu Idadhu', 'te': 'Munduku Leda Edama'},
    38: {'en': 'Keep Right', 'hi': 'Daayin Rahein', 'kn': 'Balakke Iri', 'ta': 'Valadhu Puram', 'te': 'Kudi Vaipu'},
    39: {'en': 'Keep Left', 'hi': 'Baayin Rahein', 'kn': 'Edakke Iri', 'ta': 'Idadhu Puram', 'te': 'Edama Vaipu'},
    40: {'en': 'Roundabout', 'hi': 'Gol Chakkar', 'kn': 'Vruthakaara', 'ta': 'Valai Pathai', 'te': 'Roundabout'},
    41: {'en': 'End No Passing', 'hi': 'Rok Samapt', 'kn': 'Nishedha Mugiyitu', 'ta': 'Thadai Mudivu', 'te': 'Nishedham Aipoyindi'},
    42: {'en': 'End Truck Limit', 'hi': 'Truck Rok Samapt', 'kn': 'Truck Nishedha Mugiyitu', 'ta': 'Lorry Thadai Mudivu', 'te': 'Lorry Nishedham Aipoyindi'}
}

def get_sign_text(class_id, lang='en'):
    """
    Looks up the Class ID (Number) and returns the text.
    """
    # Force convert to int in case it comes as a string or numpy int
    try:
        class_id = int(class_id)
    except:
        return str(class_id)

    if class_id in SIGN_TRANSLATIONS:
        return SIGN_TRANSLATIONS[class_id].get(lang, SIGN_TRANSLATIONS[class_id]['en'])
    
    return f"Sign {class_id}"