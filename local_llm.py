import requests

def generate_local_report(change_percent, regions):
    prompt = f"""
    Total Change: {change_percent:.2f}%

    Explain environmental impact briefly.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:latest",   # ✅ correct
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        print("STATUS:", response.status_code)
        print("TEXT:", response.text)

        return response.json().get("response", "No response")

    except Exception as e:
        print("ERROR:", e)
        return f"ERROR: {str(e)}"