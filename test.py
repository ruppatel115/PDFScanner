import openai
from openai import OpenAI


def test_api_key():
    # Get your API key
    api_key = input("Paste your OpenAI API key: ").strip()

    # Remove any extra spaces or quotes
    api_key = api_key.replace('"', '').replace("'", '')

    print(f"Testing key: {api_key[:10]}...")

    try:
        client = OpenAI(api_key=api_key)

        # Test with a very simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say just 'Hello'"}],
            max_tokens=5
        )

        print("✅ API Key is VALID!")
        print(f"Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"❌ API Key Error: {e}")
        print("\nCommon solutions:")
        print("1. Check if you added payment method at https://platform.openai.com/account/billing/overview")
        print("2. Make sure key starts with 'sk-'")
        print("3. Remove any extra spaces from copied key")
        print("4. Check your account has available credits")


if __name__ == "__main__":
    test_api_key()