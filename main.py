import re
import pickle
import numpy as np
import faiss
import json


def extract_twitter_content_and_time(text):
    results = []
    for line in text.split("\n"):
        if '．' not in line:
            continue
        # 正则表达式模式
        pattern = r'(\d+)．(.*?) *(\d{4}-\d{2}-\d{2} \d{2}:\d{2}).*'
        # 查找所有匹配
        matches = re.search(pattern, line)
        if matches:
            number = matches.group(1)
            quote = matches.group(2)
            datetime = matches.group(3)
            original_text = matches.group(0)
            results.append({
                "number": number,
                "quote": quote,
                "datetime": datetime,
                "original_text": original_text
            })
        else:
            print(f"未匹配的行: {line}")
            continue
    return results


def compute_embeddings(posts):
    from BCEmbedding import EmbeddingModel
    sentences = [post['quote'] for post in posts]
    # init embedding model
    model = EmbeddingModel(
        model_name_or_path="maidalun1020/bce-embedding-base_v1")
    # compute embeddings
    embeddings = model.encode(sentences)
    # add embeddings to posts
    for i, post in enumerate(posts):
        post['embedding'] = embeddings[i]
    return posts


def compute_similarity(posts, threshold):
    # Extract embeddings from posts
    embeddings = np.array([post['embedding'] for post in posts],
                          dtype='float32')

    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True)

    # Build the Faiss index
    d = normalized_embeddings.shape[1]  # Dimensionality of the embeddings
    index = faiss.IndexFlatIP(d)  # Inner product (dot product) index
    index.add(normalized_embeddings)  # Add embeddings to the index

    # Perform the search
    n = normalized_embeddings.shape[0]
    k = 100  # Adjust based on your needs; k should be large enough to cover potential pairs
    distances, indices = index.search(normalized_embeddings, k)

    # Filter the results
    similar_pairs = []
    for i in range(n):
        for j in range(
                1, k):  # Skip the first neighbor since it's the point itself
            if distances[i, j] > threshold and distances[i, j] < 1:
                # skip when i > j
                if i >= indices[i, j]:
                    continue
                # skip if one contains the other
                if posts[i]['quote'] in posts[indices[i, j]]['quote'] or posts[
                        indices[i, j]]['quote'] in posts[i]['quote']:
                    continue

                print(
                    f"Similar pair: {posts[i]['number']} {posts[i]['quote']} - {posts[indices[i, j]]['number']} {posts[indices[i, j]]['quote']} ({distances[i, j]})"
                )
                similar_pairs.append({
                    'source': posts[i]['number'],
                    'target': posts[indices[i, j]]['number'],
                    'value': distances[i, j]
                })

    return similar_pairs


def filter_data(posts):
    results = []
    for post in posts:

        if post['quote'] == '上传了新照片':
            continue

        cleaned_post = re.sub(r'@\w+\s*', '', post['quote'])
        if len(cleaned_post) < 10:
            continue

        if '回复' in post['original_text']:
            continue

        results.append(post)
    print("Filtered data: ", len(results))
    return results


# Custom JSON encoder to handle non-serializable objects
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":

    # # 示例输入文本
    # with open("source.txt", "r") as file:
    #     text = file.read()

    # # 提取内容和时间
    # results = extract_twitter_content_and_time(text)

    # # 打印结果
    # for result in results:
    #     print(result)

    # # 输出结果jsonl
    # with open("output.jsonl", "w") as file:
    #     for result in results:
    #         file.write(json.dumps(result, ensure_ascii=False) + "\n")

    # read from jsonl
    # with open("output.jsonl", "r") as file:
    #     data = []
    #     for line in file:
    #         # parse data to dict
    #         data.append(eval(line))
    #         # print(data)

    #     # compute embeddings
    #     compute_embeddings(data)

    #     # save data as pickle
    #     with open("output.pkl", "wb") as file:
    #         pickle.dump(data, file)

    # load from pickle
    with open("output.pkl", "rb") as file:
        posts = pickle.load(file)
        # filter data
        posts = filter_data(posts)
        similar_pairs = compute_similarity(posts, 0.7)

        # save to jsonl
        with open("similar_pairs.jsonl", "w") as file:
            for pair in similar_pairs:
                file.write(
                    json.dumps(pair, cls=NumpyEncoder, ensure_ascii=False) +
                    "\n")
