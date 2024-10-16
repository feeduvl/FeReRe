import pandas as pd


# Function to process the feedback column
def process_feedback(feedback):
    # Split at the first occurrence of ###
    parts = feedback.split("###", 1)

    # Check if we have at least two parts after the split
    if len(parts) > 1:
        new_feedback = parts[1].lstrip()  # Remove leading spaces from the remaining feedback
        # Check and remove "###" if it's at the end of the new feedback
        if new_feedback.endswith("###"):
            new_feedback = new_feedback[:-3]  # Remove the last 3 characters, which are "###"
        return parts[0], new_feedback  # Return text before ### and the modified feedback
    else:
        return "", feedback  # If ### not found, return the original feedback without changes

def process_excel(file_path, output_file_path):
    # Read the Excel file, skipping the first row
    df = pd.read_excel(file_path, skiprows=1, header=None, names=["ID", "Feedback"])

    # Create new columns for processed feedback
    df[["New Column", "Feedback"]] = df.apply(lambda row: process_feedback(row["Feedback"]), axis=1, result_type="expand")

    # Save the processed DataFrame to a new Excel file
    df.to_excel(output_file_path, index=False, header=False)

# Example usage
input_file = "input.xlsx"  # Replace with your input file path
output_file = "output.xlsx"  # Replace with your desired output file path

process_excel(input_file, output_file)
