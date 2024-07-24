import re
from transformers import BertTokenizer
import string


# Example text
text = "The pH of the NaCl solution was 7.4, measured at 25°C."

# Convert to lowercase
text = text.lower()
# Remove punctuation and numbers
text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

# Tokenization using regular expressions
tokens = re.findall(r'\w+|[^\w\s]', text)
print(tokens)

# Tokenization using BERT tokenizer (subword tokenization)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = tokenizer.tokenize(text)
print(bert_tokens)