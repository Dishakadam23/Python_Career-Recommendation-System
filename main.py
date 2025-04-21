import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import threading
import time

# Load dataset
file_path = r"C:\Users\Disha Kadam\Downloads\career_recommender.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

career_column = "If yes, then what is/was your first Job title in your current field of work? If not applicable, write NA."
skills_column = "What are your skills ? (Select multiple if necessary)"
interests_column = "What are your interests?"

required_columns = {career_column, skills_column, interests_column}
if not required_columns.issubset(df.columns):
    raise KeyError("Missing required columns in the dataset")

df.fillna("", inplace=True)
df["combined_text"] = df[interests_column] + " " + df[skills_column]
df = df.drop_duplicates(subset=[career_column])

vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
combined_tfidf = vectorizer.fit_transform(df["combined_text"])
svd = TruncatedSVD(n_components=100)
reduced_data = svd.fit_transform(combined_tfidf)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(reduced_data)

nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
nn_model.fit(scaled_data)

def recommend_careers(user_input):
    if not user_input.strip():
        return []
    input_vector = vectorizer.transform([user_input])
    input_vector_reduced = svd.transform(input_vector)
    input_vector_scaled = scaler.transform(input_vector_reduced)
    distances, indices = nn_model.kneighbors(input_vector_scaled)
    recommended_careers = df.iloc[indices[0]][career_column].values
    recommended_careers = list(dict.fromkeys([career for career in recommended_careers if career.strip() and career.upper() != "NA"]))
    return recommended_careers

# Theme Colors
themes = {
    "light": {"bg": "#f0f8f8", "fg": "#333", "frame": "#ffffff", "button": "#008080", "hover": "#00b3b3"},
    "dark": {"bg": "#1e2f33", "fg": "#eeeeee", "frame": "#2f4f4f", "button": "#00cccc", "hover": "#00aaaa"},
}
current_theme = "light"

# GUI Setup
app = tk.Tk()
app.title("Career Recommendation System")
app.attributes('-fullscreen', True)
app.configure(bg=themes[current_theme]["bg"])

style = ttk.Style()
style.theme_use("clam")

# Tooltip Helper
def create_tooltip(widget, text):
    tooltip = tk.Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    tooltip_label = tk.Label(tooltip, text=text, background="lightyellow", relief="solid", borderwidth=1, font=("Arial", 10))
    tooltip_label.pack()

    def enter(event):
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 20
        tooltip.geometry(f"+{x}+{y}")
        tooltip.deiconify()

    def leave(event):
        tooltip.withdraw()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

# Hover effect for buttons
def on_enter(e):
    e.widget["background"] = themes[current_theme]["hover"]

def on_leave(e):
    e.widget["background"] = themes[current_theme]["button"]

# Animation function
def animate_label(label):
    def blink():
        colors = [themes[current_theme]["button"], themes[current_theme]["hover"]]
        i = 0
        while True:
            label.config(fg=colors[i % len(colors)])
            i += 1
            time.sleep(0.75)
    thread = threading.Thread(target=blink, daemon=True)
    thread.start()

# Toggle Theme
def toggle_theme():
    global current_theme
    current_theme = "dark" if current_theme == "light" else "light"
    app.configure(bg=themes[current_theme]["bg"])
    main_frame.configure(bg=themes[current_theme]["frame"])
    header.configure(bg=themes[current_theme]["frame"], fg=themes[current_theme]["button"])
    result_box.configure(bg=themes[current_theme]["frame"], fg=themes[current_theme]["fg"])

# Overlay Frame
main_frame = tk.Frame(app, bg=themes[current_theme]["frame"], padx=40, pady=40, bd=2, relief="groove")
main_frame.place(relx=0.5, rely=0.5, anchor="center")

# Title
header = tk.Label(main_frame, text="\ud83d\udcc8 Career Recommendation System", font=("Helvetica", 26, "bold"), fg=themes[current_theme]["button"], bg=themes[current_theme]["frame"])
header.grid(row=0, column=0, columnspan=2, pady=(10, 30))
animate_label(header)

# Interests
lbl_interest = tk.Label(main_frame, text="\ud83c\udf1f Select Your Interests:", font=("Arial", 14), bg=themes[current_theme]["frame"], fg=themes[current_theme]["fg"])
lbl_interest.grid(row=1, column=0, sticky="e", padx=10, pady=10)
interest_entry = ttk.Combobox(main_frame, values=[x for x in df[interests_column].unique() if x], width=45)
interest_entry.grid(row=1, column=1, pady=10, padx=10)
create_tooltip(interest_entry, "Choose your interest areas")

# Skills
lbl_skill = tk.Label(main_frame, text="\ud83d\udcaa Select Your Skills:", font=("Arial", 14), bg=themes[current_theme]["frame"], fg=themes[current_theme]["fg"])
lbl_skill.grid(row=2, column=0, sticky="e", padx=10, pady=10)
skill_entry = ttk.Combobox(main_frame, values=[x for x in df[skills_column].unique() if x], width=45)
skill_entry.grid(row=2, column=1, pady=10, padx=10)
create_tooltip(skill_entry, "Mention skills you have")

# Recommendation function
def get_recommendation():
    user_input = f"{interest_entry.get()} {skill_entry.get()}".strip()
    if not user_input:
        messagebox.showerror("Error", "Please select your interests and skills!")
        return
    loading_label.config(text="\u23f3 Finding careers...")
    app.update_idletasks()
    time.sleep(0.5)
    recommendations = recommend_careers(user_input)
    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    if not recommendations:
        result_box.insert(tk.END, "No suitable career recommendations found.")
    else:
        result_text = "\n\n".join([f"\ud83d\udcbc {i+1}. {career}" for i, career in enumerate(recommendations)])
        result_box.insert(tk.END, f"\ud83d\udcc4 Recommended Careers:\n\n{result_text}")
    result_box.config(state="disabled")
    loading_label.config(text="")

# Buttons
recommend_btn = tk.Button(main_frame, text="\ud83d\udcca Get Career Recommendation", font=("Arial", 14, "bold"), bg=themes[current_theme]["button"], fg="white", padx=10, pady=5, command=get_recommendation)
recommend_btn.grid(row=3, column=0, columnspan=2, pady=20)
recommend_btn.bind("<Enter>", on_enter)
recommend_btn.bind("<Leave>", on_leave)

# Result Box
result_box = tk.Text(main_frame, height=10, width=65, font=("Arial", 12), wrap="word", bg=themes[current_theme]["frame"], fg=themes[current_theme]["fg"])
result_box.grid(row=4, column=0, columnspan=2, pady=10, padx=10)
result_box.config(state="disabled")

# Loading Label
loading_label = tk.Label(main_frame, text="", font=("Arial", 12, "italic"), bg=themes[current_theme]["frame"], fg="gray")
loading_label.grid(row=5, column=0, columnspan=2)

# Theme Toggle Button
toggle_btn = tk.Button(app, text="\ud83c\udf1e Toggle Theme", font=("Arial", 12), bg=themes[current_theme]["button"], fg="white", command=toggle_theme)
toggle_btn.place(x=20, y=60)
toggle_btn.bind("<Enter>", on_enter)
toggle_btn.bind("<Leave>", on_leave)

# Exit Button
exit_btn = tk.Button(app, text="❌ Exit", font=("Arial", 12), bg="#ff4d4d", fg="white", command=app.destroy)
exit_btn.place(x=20, y=20)

app.mainloop()
