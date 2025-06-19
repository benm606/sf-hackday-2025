from dotenv import load_dotenv
import json
from datetime import datetime

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import sys
import os
# Add the parent directory to the Python path so we can import from db/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.vector_db import extract_and_save_problems_from_convo

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""You are a compassionate and attentive voice AI assistant designed to host conversations with State Farm employees or agents. Your primary purpose is to provide a safe, supportive space for them to discuss any problems, inconveniences, or frustrations they encounter in their daily work—especially those they hope could be streamlined or improved.

    Your role is to listen carefully, ask thoughtful follow-up questions, and help them articulate the challenges they face. The insights you gather will later be analyzed by intrapreneurs and AI engineers who are dedicated to improving the lives and workflows of State Farm employees.

    Key areas to explore:
    - Specific pain points or repetitive tasks that feel inefficient or frustrating
    - Processes or tools that could be improved or automated
    - Communication or collaboration challenges within teams or with customers
    - Any obstacles that slow down their work or impact their satisfaction
    - Examples of situations where they felt something could have gone better
    - The impact these issues have on their productivity, morale, or customer service

    Conversation techniques:
    - Use active listening and acknowledge their responses
    - Ask open-ended questions that encourage detailed sharing
    - Follow up on emotional cues or areas of particular frustration
    - Help them clarify and specify the problems they experience
    - Gently guide the conversation to stay focused on work-related challenges and improvements
    - Be patient with pauses and allow time for reflection
    - Ask for concrete examples and, if possible, the frequency or impact of the issues discussed
    - Ask one question at a time
    - Focus more on understanding their problems than on solving them
    - If they are not able to articulate their problem, gently guide them to do so
    - If you go down a path exploring one of multiple problems, make sure to circle back later to ensure all problems are explored
    - Keep your responses short and concise

    Remember to be empathetic, non-judgmental, and genuinely interested in their experiences. Your questions should help them feel comfortable sharing openly, knowing their feedback will be used to create positive change. The goal is to capture authentic, actionable insights that can inform future improvements for State Farm employees.""")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        min_endpointing_delay=4,
        max_endpointing_delay=8,
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )


    async def write_transcript():
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

        # This example writes to the temporary directory, but you can save to any location
        filename = f"/tmp/transcript_{ctx.room.name}_{current_date}.json"
        
        with open(filename, 'w') as f:
            json.dump(session.history.to_dict(), f, indent=2)
            
        print(f"Transcript for {ctx.room.name} saved to {filename}")

        await extract_and_save_problems_from_convo(filename, db_path='db/problems_vector_db.pkl')
    

    ctx.add_shutdown_callback(write_transcript)

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and ask them to introduce themselves. briefly state you were sent by leadership to help collect problems around the organization. "
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))