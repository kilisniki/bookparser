def merge_vocabularies(vocabulary, new_word_map):
    for word, data in new_word_map.items():
        if word in vocabulary:
            # Суммируем count
            vocabulary[word]['count'] += data['count']

            # Объединяем уникальные формы слов
            vocabulary[word]['forms'] = list(set(vocabulary[word]['forms']).union(set(data['forms'])))

            # Объединяем предложения для каждой формы
            chosen_sentences = []
            for form in vocabulary[word]['forms']:
                # Ищем предложения в старом словаре
                for sentence in vocabulary[word]['chosen_sentences']:
                    if form in sentence:
                        chosen_sentences.append(sentence)
                # Ищем предложения в новом словаре
                for sentence in data['chosen_sentences']:
                    if form in sentence:
                        chosen_sentences.append(sentence)

            vocabulary[word]['chosen_sentences'] = chosen_sentences
        else:
            # Добавляем новое слово в vocabulary
            vocabulary[word] = data
    return vocabulary
