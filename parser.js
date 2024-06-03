const fs = require('fs');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

// Функция для извлечения слов из текста
function extractWordsFromText(text) {
  // Регулярное выражение для удаления знаков препинания, но сохранения апострофов и чисел
  const cleanedText = text.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"“”‘’]/g, '');

  // Разделяем текст на слова по пробелам
  const words = cleanedText.split(/\s+/);

  // Удаляем пустые строки, которые могут появиться из-за нескольких пробелов
  return words.filter(word => word.length > 0);
}

// Функция для подсчета уникальных слов и их частоты
function countUniqueWords(words) {
  const wordCount = {};

  words.forEach(word => {
    const lowerCaseWord = word.toLowerCase();
    if (wordCount[lowerCaseWord]) {
      wordCount[lowerCaseWord]++;
    } else {
      wordCount[lowerCaseWord] = 1;
    }
  });

  return wordCount;
}

// Функция для записи слов и их частоты в CSV-файл
async function exportWordCountsToCsv(wordCounts, filePath) {
  const csvWriter = createCsvWriter({
    path: filePath,
    header: [
      { id: 'word', title: 'Word' },
      { id: 'count', title: 'Count' }
    ]
  });

  const records = Object.keys(wordCounts).map(word => ({
    word,
    count: wordCounts[word]
  }));

  try {
    await csvWriter.writeRecords(records);
    console.log('CSV file written successfully');
  } catch (error) {
    console.error('Error writing CSV file:', error);
  }
}

// Пример использования функций
(async () => {
  console.log('start')
  const bookText = (await fs.promises.readFile('./thewitcher.txt')).toString();
  console.log('book is read');
  // const bookText = "Hello, world! This is a test. Let's remove punctuation, and special characters. Hello world!";
  const words = extractWordsFromText(bookText);
  console.log('words are extracted');
  const wordCounts = countUniqueWords(words);
  console.log('words are counted');

  await exportWordCountsToCsv(wordCounts, 'word_counts.csv');
  console.log('exported!')
})();
