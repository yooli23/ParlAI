from flask import Flask, request
from parlai.core.agents import create_agent_from_model_file


app = Flask(__name__)
blender = create_agent_from_model_file("zoo:blender/blender_90M/model")


@app.route('/blender', methods=['POST'])
def index():    
    history = request.json['history']    
    blender.reset()    
    i = 0
    while history:
        utt = history.pop(0)
        if i % 2 == 0:
            blender.observe({'text': utt, 'episode_done': False})
        else:
            blender.self_observe({'text': utt})
        i += 1
    response = blender.act()
    result = response['text'] \
            .replace(" '", "'") \
            .replace("' ", "'") \
            .replace(" .", "") \
            .replace(" ?", "?") \
            .replace(" ,", ",")    
    return {'result': result}


if __name__ == '__main__':
    app.run(port=3000)