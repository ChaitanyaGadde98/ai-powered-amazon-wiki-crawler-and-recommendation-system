from flask import Flask, render_template, request, jsonify
from utils import amazon_scraper, wikipedia_scraper, get_similarity_scores

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
    search_term = request.args.get('search')
    # response_data = {'message': 'Received search: ' + search}
    # return jsonify(response_data)

    amazon_results = amazon_scraper(search_term)
    print(amazon_results)
    wikipedia_result = wikipedia_scraper(search_term)

    amazon_results = get_similarity_scores(amazon_results, wikipedia_result)


    response = {"wiki": wikipedia_result,
                "amazon": amazon_results}

    print(response)
    return jsonify(response)





if __name__ == '__main__':
    app.run(debug=True, port=1234)
