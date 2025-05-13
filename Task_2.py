import speech_recognition as sr
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("🎙️ Speak something...")
    audio = recognizer.listen(source)

    try:
        print("📝 Transcribing...")
        text = recognizer.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        print("❌ Could not understand the audio")
    except sr.RequestError as e:
        print(f"❌ Could not request results; {e}")
