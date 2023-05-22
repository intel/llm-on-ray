# Ray-chatbot
- Two step: Ray serve provides services, Gradio is responsible for the webui.
    - python start_ray_serve.py
    - python start_ui.py
    - open http://10.165.54.55:8081/

- Scaling Gradio app with Ray Serve (GradioIngress)
    - serve run Ray_gradio.chat_bot:app -h 10.165.54.55 -p 8000
    - open http://10.165.54.55:8000/