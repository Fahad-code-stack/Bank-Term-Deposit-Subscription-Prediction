import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

class BankDepositPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bank Term Deposit Predictor")
        self.root.geometry("500x600")
        self.root.configure(bg="#f4f6f9")
        
        # Load model and encoders
        try:
            self.model = joblib.load("bank_model.pkl")
            self.encoders = joblib.load("encoders.pkl")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or encoders:\n{str(e)}")
            self.root.destroy()
            return
        
        self.setup_ui()
        
    def setup_ui(self):
        # Styling
        style = ttk.Style()
        # Use clam theme for a more modern look across platforms
        style.theme_use('clam')
        style.configure("TLabel", background="#ffffff", font=("Helvetica", 11))
        style.configure("TButton", font=("Helvetica", 12, "bold"), background="#007bff", foreground="white", padding=5)
        style.map("TButton", background=[("active", "#0056b3")])
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), background="#f4f6f9", foreground="#333333")
        
        # Header
        header = ttk.Label(self.root, text="Bank Term Deposit Predictor", style="Header.TLabel")
        header.pack(pady=20)
        
        # Main Frame
        frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief=tk.RIDGE)
        frame.pack(padx=30, pady=10, fill=tk.BOTH, expand=True)
        
        # Form grid configuration
        frame.columnconfigure(0, weight=1, pad=10)
        frame.columnconfigure(1, weight=2, pad=10)
        
        # Field variables
        self.age_var = tk.StringVar()
        self.balance_var = tk.StringVar()
        self.job_var = tk.StringVar()
        self.marital_var = tk.StringVar()
        self.education_var = tk.StringVar()
        self.default_var = tk.StringVar()
        self.housing_var = tk.StringVar()
        self.loan_var = tk.StringVar()
        
        row = 0
        
        # Age
        ttk.Label(frame, text="Age:").grid(row=row, column=0, sticky=tk.W, pady=10, padx=15)
        self.age_entry = ttk.Entry(frame, textvariable=self.age_var, font=("Helvetica", 11))
        self.age_entry.grid(row=row, column=1, sticky=tk.EW, pady=10, padx=15)
        row += 1
        
        # Balance
        ttk.Label(frame, text="Balance:").grid(row=row, column=0, sticky=tk.W, pady=10, padx=15)
        self.balance_entry = ttk.Entry(frame, textvariable=self.balance_var, font=("Helvetica", 11))
        self.balance_entry.grid(row=row, column=1, sticky=tk.EW, pady=10, padx=15)
        row += 1
        
        # Helper to create dropdowns
        def create_dropdown(label_text, var, encoder_key, r):
            ttk.Label(frame, text=label_text).grid(row=r, column=0, sticky=tk.W, pady=10, padx=15)
            # Retrieve classes from the LabelEncoder dynamically
            values = list(self.encoders[encoder_key].classes_)
            cb = ttk.Combobox(frame, textvariable=var, values=values, state="readonly", font=("Helvetica", 11))
            if values:
                cb.current(0)
            cb.grid(row=r, column=1, sticky=tk.EW, pady=10, padx=15)
            return r + 1

        row = create_dropdown("Job:", self.job_var, "job", row)
        row = create_dropdown("Marital Status:", self.marital_var, "marital", row)
        row = create_dropdown("Education:", self.education_var, "education", row)
        row = create_dropdown("Credit Default:", self.default_var, "default", row)
        row = create_dropdown("Housing Loan:", self.housing_var, "housing", row)
        row = create_dropdown("Personal Loan:", self.loan_var, "loan", row)
        
        # Predict Button
        predict_btn = ttk.Button(self.root, text="Predict", command=self.predict)
        predict_btn.pack(pady=15, ipadx=20)
        
        # Result Label
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"), bg="#f4f6f9")
        self.result_label.pack(pady=10)

    def predict(self):
        # 1. Validation
        age_str = self.age_var.get().strip()
        balance_str = self.balance_var.get().strip()
        
        if not age_str or not balance_str:
            messagebox.showwarning("Validation Error", "Age and Balance must not be empty.")
            return
        
        try:
            age = int(age_str)
            if age <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Validation Error", "Age must be a positive integer.")
            return
            
        try:
            balance = float(balance_str)
        except ValueError:
            messagebox.showwarning("Validation Error", "Balance must be a valid number.")
            return

        # 2. Gather inputs in exact order
        sample = {
            "age": age,
            "job": self.job_var.get(),
            "marital": self.marital_var.get(),
            "education": self.education_var.get(),
            "default": self.default_var.get(),
            "balance": balance,
            "housing": self.housing_var.get(),
            "loan": self.loan_var.get()
        }
        
        # 3. Transform inputs using encoders
        try:
            for col in self.encoders:
                if col != "y" and col in sample:
                    # Encoders expect a list-like input for transform
                    sample[col] = self.encoders[col].transform([sample[col]])[0]
        except Exception as e:
            messagebox.showerror("Error", f"Error encoding categorical features:\n{str(e)}")
            return
            
        # 4. Predict
        try:
            sample_df = pd.DataFrame([sample])
            prediction = self.model.predict(sample_df)[0]
            
            if prediction == 1:
                self.result_label.config(text="Customer will subscribe to the term deposit", fg="#28a745") # Green
            else:
                self.result_label.config(text="Customer will NOT subscribe to the term deposit", fg="#dc3545") # Red
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BankDepositPredictorApp(root)
    root.mainloop()
