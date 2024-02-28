from jira import JIRA
import pandas as pd
import re

def extractIssuesFromJira(path, filterPrefix=False, filterSoftwareNames=None):
    print("Extracting Jira Issues")
    # Jira server setup
    jira_url = "https://jira-se.ifi.uni-heidelberg.de"
    username = input("Username: ")
    password = input("Password: ")

    # Connect to Jira
    jira = JIRA(jira_url, basic_auth=(username, password))

    # Define the JQL query to fetch issues of specific types from the project
    jql_query = 'project = KOMOOT AND issuetype in ("System Function", "Workspace", "User Subtask")'

    # Fetch issues using JQL
    issues = jira.search_issues(jql_query, maxResults=False)

    # Prepare data for Excel
    data = []
    for issue in issues:
        key = issue.key
        summary = issue.fields.summary
        description = issue.fields.description

        if filterPrefix:
            # Pattern to match "SF"
            sf_pattern = r"SF"
            # Pattern to match "W" followed by a decimal number (with optional decimal part) or whole number
            w_pattern = r"W\d+(\.\d+)?"
            # Pattern to match "UT" followed by a number, "S", and another number
            ut_pattern = r"UT\d+S\d+"
            # Combine all patterns into one using the | operator, which stands for "OR"
            combined_pattern = f"{sf_pattern}|{w_pattern}|{ut_pattern}"
            # Replace occurrences of the combined pattern with an empty string
            summary = re.sub(combined_pattern, '', summary)
            summary = summary.replace(":","")
        if filterSoftwareNames != None:
            # Escape each string in filterName to handle strings that might contain regex special characters
            escaped_filterName = [re.escape(name) for name in filterSoftwareNames]

            # Join the escaped strings into a single pattern with '|' as OR operator
            pattern = '|'.join(escaped_filterName)

            # Replace occurrences of the pattern with "software"
            summary = re.sub(pattern, "software", summary)
            description = re.sub(pattern, "software", description)

        data.append([key, summary + ": " + description])

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Write DataFrame to Excel file
    if filterPrefix:
        excel_path = path+'jira_issues_noprefix.xlsx'
        if filterSoftwareNames != None:
            excel_path = path+'jira_issues_namesfiltered_noprefix.xlsx'
    else:
        if filterSoftwareNames != None:
            excel_path = path+'jira_issues_namesfiltered.xlsx'
        excel_path = path+'jira_issues.xlsx'
    df.to_excel(excel_path, index=False, header=False, engine='openpyxl')

    print(f"Excel file has been created at: {excel_path}")
