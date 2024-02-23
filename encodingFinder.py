import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        # Read some bytes from the file; you might need to adjust the byte count
        raw_data = file.read(10000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']  # Represents how confident chardet is about its guess
        return encoding, confidence

file_path = 'input.csv'  # Replace this with your file's path
encoding, confidence = detect_encoding(file_path)

print(f"Detected Encoding: {encoding} with {confidence*100:.2f}% confidence")
