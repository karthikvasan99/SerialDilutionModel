with open("plot_dopri5_bifurcation_proof.py", "r") as f:
    text = f.read()

# Replace \ge with \geq
text = text.replace(r"$R \ge 0$", r"$R \geq 0$")
text = text.replace(r"$N_{\max}$", r"$N_{max}$")
text = text.replace(r"$N_{\min}$", r"$N_{min}$")

with open("plot_dopri5_bifurcation_proof.py", "w") as f:
    f.write(text)

print("Replacement complete.")
