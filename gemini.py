from time import sleep

from google import genai



google_api_key = "AQ.Ab8RN6Lx-dd2skM8Y0yyg-uiEtV-QSmBkZQuCkRgj1blrm3Cmg"
#


client = genai.Client(api_key=google_api_key)


for i in range(10):
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input="Explain how AI works in a few words"
    )
    print(interaction.output_text)
    sleep(1)
