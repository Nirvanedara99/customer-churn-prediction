import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    # Default threshold
    threshold = 0.4  

    # If user passes a threshold as command-line argument, use it
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            print("Invalid threshold provided. Using default 0.4.")

    # Load predictions
    df = pd.read_csv("predictions.csv")

    # Find high-risk customers
    high_risk = df[df["Churn_Probability"] > threshold]

    print(f"Total customers: {len(df)}")
    print(f"High-risk customers (prob > {threshold}): {len(high_risk)}\n")

    print("Sample of high-risk customers:")
    print(high_risk.head(), "\n")

    # Save to CSV
    high_risk.to_csv("high_risk_customers.csv", index=False)
    print("High-risk customers saved to high_risk_customers.csv")

    # ----------------------
    # Plot churn probability distribution
    plt.hist(df["Churn_Probability"], bins=10, edgecolor="black")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold}")
    plt.xlabel("Churn Probability")
    plt.ylabel("Number of Customers")
    plt.title("Churn Probability Distribution")
    plt.legend()
    plt.show()

    # ----------------------
    # Plot high-risk customers (horizontal bar chart)
    if not high_risk.empty:
        plt.barh(high_risk["customerID"], high_risk["Churn_Probability"])
        plt.xlabel("Churn Probability")
        plt.ylabel("Customer ID")
        plt.title("High-Risk Customers")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
