import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

medical_report = """
Patient Student, aged 21 years, was admitted to Jaslok Hospital with
complaints of fever and chest pain. He was diagnosed with pneumonia and prescribed
Azithromycin 500 mg for 5 days.
"""

doc = nlp(medical_report)

print("Named Entities:")
print("----------------")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)

entities = [(ent.text, ent.label_) for ent in doc.ents]
df = pd.DataFrame(
    entities,
    columns=["Entity", "Entity Type"]
)

print("DataFrame:")
print(df)

df.to_csv("Medical_Entities.csv", index=False)
print("\nMedical_Entities.csv saved successfully!")

print("\nMedical Information:")
medical_terms = ["pneumonia", "Azithromycin", "fever", "chest pain"]
for term in medical_terms:
    if term.lower() in medical_report.lower():
        print(term)
