from flask import Flask, render_template, session, request, jsonify
import json
import os
import time

from raw_blender import RawBlenderBot, RAW_BLENDER_PATH

# Init the server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
chatbot = RawBlenderBot()
print("loading raw blender model")
chatbot.load_response_generator(RAW_BLENDER_PATH, gpu_num=0)

manager = {}

def get_chat_input(sid):
    if sid in manager.keys():
        history_with_entities = manager[sid]
    else:
        history_with_entities = []
        manager[sid] = history_with_entities
    return history_with_entities

def set_chat_history(sid, history_with_entities, user_text, response):
    if sid not in manager.keys():
        print("Raw blender ERROR! sid is not in manager!")
    else:
        if history_with_entities:
            history_with_entities.append(user_text)
        history_with_entities.append(response)
        manager[sid] = history_with_entities

@app.route("/raw_blender", methods=['POST'])
def get_Response():
    """ Get response from the raw blender chatbot."""
    sid = request.json.get('sid')
    context = request.json.get('context')
    user_text = request.json.get('user_text')
    print("get message, sid={}".format(sid))
    history_with_entities = get_chat_input(sid)
    response, end_chat = chatbot.chat(history_with_entities, user_text)
    set_chat_history(sid, history_with_entities, user_text, response)

    return jsonify({"response": response, "end_chat": end_chat})

@app.route("/raw_blender_chitchat", methods=['POST'])
def get_chitchat_Response():
    """ Get response from the raw blender chichat."""
    history_with_entities = request.json.get('history_with_entities')
    user_text = request.json.get('user_text')

    response, end_chat = chatbot.chat(history_with_entities, user_text)

    return jsonify({"response": response})

if __name__ == '__main__':
    """ Run the app. """
    # socketio.run(app, port=3322)
    app.run(host='0.0.0.0', port=6779)


