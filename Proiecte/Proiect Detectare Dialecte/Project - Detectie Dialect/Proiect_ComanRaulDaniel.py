import tkinter as tk
from tkinter import scrolledtext, messagebox
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


def antreneaza_model_din_fisier():
    x_date = []
    y_etichete = []
    nume_fisier = "dialecte.txt"

    if os.path.exists(nume_fisier):
        with open(nume_fisier, "r", encoding="utf-8") as f:
            for linie in f:
                if "," in linie:
                    parts = linie.strip().rsplit(",", 1)
                    if len(parts) == 2:
                        x_date.append(parts[0].lower())
                        y_etichete.append(parts[1])

    if not x_date:
        return None

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(x_date, y_etichete)
    return model


model_ai = antreneaza_model_din_fisier()


def detecteaza_dialect_ml():
    if model_ai is None:
        messagebox.showerror("Eroare", "Fișierul dialecte.txt e gol sau lipsește!")
        return

    text_utilizator = text_input.get("1.0", tk.END).strip().lower()
    if not text_utilizator:
        return

    probele = model_ai.predict_proba([text_utilizator])[0]
    clase = model_ai.classes_
    rezultate = sorted(zip(clase, probele), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 40)
    print(f"INPUT: {text_utilizator}")
    for regiune, prob in rezultate:
        print(f"{regiune}: {prob * 100:.2f}%")

    label_rezultat.config(text="ANALIZĂ PROBABILISTICĂ (ML):", fg="#2c3e50")
    text_afisat = ""
    for regiune, prob in rezultate:
        if prob > 0.01:
            text_afisat += f"{regiune:.<15} {prob * 100:>5.1f}%\n"

    procente_label.config(text=text_afisat)


def sterge_text():
    text_input.delete("1.0", tk.END)
    label_rezultat.config(text="Aștept text pentru...", fg="#7f8c8d")
    procente_label.config(text="")


root = tk.Tk()
root.title("Detector Dialecte Românești")
root.geometry("600x600")
root.configure(bg="#f5f6fa")

header = tk.Frame(root, bg="#1e272e", height=80)
header.pack(fill="x")
tk.Label(header, text="DETECTOR DE DIALECTE", font=("Consolas", 14, "bold"),
         bg="#1e272e", fg="#05c46b").pack(pady=20)

main_frame = tk.Frame(root, bg="#f5f6fa")
main_frame.pack(pady=20, padx=20)

text_input = scrolledtext.ScrolledText(main_frame, width=50, height=8, font=("Segoe UI", 11))
text_input.pack(pady=10)

tk.Button(main_frame, text="DETECTEAZĂ DIALECTUL", command=detecteaza_dialect_ml,
          bg="#05c46b", fg="white", font=("Segoe UI", 10, "bold"), padx=20, pady=10).pack(pady=5)

tk.Button(main_frame, text="ȘTERGE", command=sterge_text, bg="#ff3f34", fg="white").pack()

label_rezultat = tk.Label(main_frame, text="Aștept text pentru...", font=("Segoe UI", 12, "bold"),
                          bg="#f5f6fa", fg="#7f8c8d")
label_rezultat.pack(pady=20)

procente_label = tk.Label(main_frame, text="", font=("Courier New", 16, "bold"),
                          bg="#f5f6fa", fg="#1e272e", justify="left")
procente_label.pack()

root.mainloop()