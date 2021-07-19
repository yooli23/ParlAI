from flask import Flask, render_template, session, request, jsonify
import json
import os
import time

from chatbot import Chatbot, RESPONSE_GENERATOR_PATH

# Init the server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
chatbot = Chatbot()
print("loading response generator model")
chatbot.load_response_generator(RESPONSE_GENERATOR_PATH, gpu_num=0)

@app.route("/mid_blender", methods=['POST'])
def get_Response():
    """ Get response from the movie recommendation chatbot."""
    history_with_entities = request.json.get('history_with_entities')
    user_text = request.json.get('user_text')
    entity_condition = request.json.get('entity_condition')

    response = chatbot.chat(history_with_entities, user_text, entity_condition)

    return jsonify({"response": response})

if __name__ == '__main__':
    """ Run the app. """
    # socketio.run(app, port=3322)
    app.run(host='0.0.0.0', port=6778)


