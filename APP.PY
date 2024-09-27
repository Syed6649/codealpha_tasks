import tkinter as tk
from tkinter import ttk, messagebox
from googletrans import Translator, LANGUAGES

# Initialize the Translator
translator = Translator()

# Function to Translate Text
def translate_text():
    try:
        input_text = input_textbox.get("1.0", "end-1c")
        src_lang = src_lang_combobox.get()
        dest_lang = dest_lang_combobox.get()

        if not input_text.strip():
            messagebox.showwarning("Warning", "Please enter text to translate.")
            return

        # Perform the translation
        translated = translator.translate(input_text, src=src_lang, dest=dest_lang)
        output_textbox.delete("1.0", tk.END)
        output_textbox.insert(tk.END, translated.text)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the Main Window
window = tk.Tk()
window.title("Language Translator")
window.geometry("600x400")

# Create Widgets
input_label = tk.Label(window, text="Enter text to translate:")
input_label.pack(pady=10)

input_textbox = tk.Text(window, height=10, width=60)
input_textbox.pack(pady=10)

frame = tk.Frame(window)
frame.pack(pady=10)

src_lang_label = tk.Label(frame, text="Source Language:")
src_lang_label.grid(row=0, column=0, padx=5)

src_lang_combobox = ttk.Combobox(frame, values=list(LANGUAGES.values()), width=15)
src_lang_combobox.grid(row=0, column=1, padx=5)
src_lang_combobox.set("english")

dest_lang_label = tk.Label(frame, text="Destination Language:")
dest_lang_label.grid(row=0, column=2, padx=5)

dest_lang_combobox = ttk.Combobox(frame, values=list(LANGUAGES.values()), width=15)
dest_lang_combobox.grid(row=0, column=3, padx=5)
dest_lang_combobox.set("spanish")

translate_button = tk.Button(window, text="Translate", command=translate_text)
translate_button.pack(pady=10)

output_label = tk.Label(window, text="Translated text:")
output_label.pack(pady=10)

output_textbox = tk.Text(window, height=10, width=60)
output_textbox.pack(pady=10)

# Start the GUI Event Loop
window.mainloop()

