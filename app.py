import os
import gradio as gr
from google import genai

# Initialize Gemini Client using your secret environment variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are an AI assistant for rizwansict.com (founded by Md. Rizwanul Haque), helping HSC candidates and their parents with ICT, Biology, and exam preparation.

GUIDELINES FOR YOUR RESPONSES:
1. Answer the user's question directly, clearly, and naturally like a helpful human tutor or peer.
2. Keep responses concise and easy to read on mobile devices.
3. DO NOT force or push registration in every single reply or right at the beginning. Help the user first.
4. Only suggest visiting https://rizwansict.com/ to register for free learning resources when relevant or after providing a helpful answer.

For calling/WhatsApp chat:
Use: 8801711825681

For eMail:
Use: rizwansict.rajshahi@gmail.com or connect@rizwansict.com

For Address/Location:
Use: Bulu Vila (dotola), South to Kadirganj Darikharbana Greater Road Jame Masjid, Kadirganj, Rajshahi, Bangladesh.

For Online Admission:
Visite: https://rizwansict.com/register

Batch Schedule HSC ICT:
Sat-Mon-Wed: 7am-8am, 8am-9am, 6pm-7pm, 7pm-8pm
Sun-Tue-Thu: 7am-8am, 8am-9am, 6pm-7pm, 7pm-8pm

Batch Schedule HSC Biology:
Sat-Mon-Wed: 4pm-5pm, 5pm-6pm
Sun-Tue-Thu: 4pm-5pm, 5pm-6pm

Exam Day: Friday, Time would be disclosed later.
"""

MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

def respond(message, history):
    full_prompt = f"{SYSTEM_PROMPT}\n\n"
    
    if history:
        for turn in history:
            if isinstance(turn, (tuple, list)):
                user_text, bot_text = turn[0], turn[1]
            elif isinstance(turn, dict):
                user_text = turn.get("user") or turn.get("content", "")
                bot_text = turn.get("bot") or turn.get("metadata", "")
            else:
                continue

            if user_text:
                full_prompt += f"User: {user_text}\n"
            if bot_text:
                full_prompt += f"Assistant: {bot_text}\n"

    full_prompt += f"User: {message}\nAssistant:"

    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            if response and response.text:
                return response.text
        except Exception:
            continue

    return "Rizwan's ICT Helpline is not available right now! Please try asking your question again in a moment, feel free to check out our study resources directly on rizwansict.com! or for urgency click to <a href='https://wa.me/8801711825681'>WhatsApp Chat</a>"

demo = gr.ChatInterface(
    fn=respond,
    title="Rizwan's ICT - SSC/HSC Prep Assistant",
    description="Ask anything about HSC ICT, Biology, or platform features!",
    textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7)
)

if __name__ == "__main__":
    # Render assigns dynamic port numbers via $PORT environment variable
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
