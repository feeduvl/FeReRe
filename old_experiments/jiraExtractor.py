import networkx as nx
from jira import JIRA
import pandas as pd
import re

def extractIssuesFromJira(path, filterPrefix=False, filterSoftwareNames=None, project="KOMOOT"):
    print("Extracting Jira Issues")
    # Jira server setup
    jira_url = "INSERT URL"
    username = input("Username: ")
    password = input("Password: ")

    # Connect to Jira
    jira = JIRA(jira_url, basic_auth=(username, password))

    # Define the JQL query to fetch issues of specific types from the project
    jql_query = f'project = {project} AND issuetype in ("System Function", "Workspace", "User Subtask")'

    # Fetch issues using JQL
    issues = jira.search_issues(jql_query, maxResults=False)

    # Prepare data for Excel
    data = []
    for issue in issues:
        key = issue.key
        summary = issue.fields.summary
        description = issue.fields.description

        summary,description = applyFilters(summary,description,filterPrefix,filterSoftwareNames)
        print(summary)
        print(description)
        data.append([key, summary + ": " + description])

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Write DataFrame to Excel file
    if filterPrefix:
        excel_path = path+'jira_issues_noprefix_old.xlsx'
        if filterSoftwareNames != None:
            excel_path = path+'jira_issues_namesfiltered_noprefix.xlsx'
    else:
        if filterSoftwareNames != None:
            excel_path = path+'jira_issues_namesfiltered.xlsx'
        excel_path = path+'jira_issues.xlsx'
    df.to_excel(excel_path, index=False, header=False, engine='openpyxl')

    print(f"Excel file has been created at: {excel_path}")

def extractIssuesFromJiraWithLinks(path, filterPrefix=False, filterSoftwareNames=None, project="KOMOOT"):
    print("Extracting Jira Issues")
    # Jira server setup
    jira_url = "https://jira-se.ifi.uni-heidelberg.de"
    username = input("Username: ")
    password = input("Password: ")

    # Connect to Jira
    jira = JIRA(jira_url, basic_auth=(username, password))

    # Define the JQL query to fetch issues of specific types from the project
    jql_query = f'project = {project} AND issuetype in ("System Function", "Workspace", "User Subtask")'

    # Fetch issues using JQL
    issues = jira.search_issues(jql_query, maxResults=False, fields="summary,description,issuelinks")

    # Prepare data for Excel
    data = []
    for issue in issues:
        key = issue.key
        summary = issue.fields.summary
        description = issue.fields.description
        # Fetch linked issues
        linked_issues = []
        for link in issue.fields.issuelinks:
            if hasattr(link, "outwardIssue"):
                linked_issues.append(link.outwardIssue.key)
            elif hasattr(link, "inwardIssue"):
                linked_issues.append(link.inwardIssue.key)

        # Convert list of linked issues to a string
        linked_issues_str = ', '.join(linked_issues)
        summary,description = applyFilters(summary,description,filterPrefix,filterSoftwareNames)

        data.append([key, summary, description, linked_issues_str])

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Write DataFrame to Excel file
    if filterPrefix:
        excel_path = path+'jira_issues_Linked_noprefix.xlsx'
        if filterSoftwareNames != None:
            excel_path = path+'jira_issues_linked_namesfiltered_noprefix.xlsx'
    else:
        if filterSoftwareNames != None:
            excel_path = path+'jira_issues_linked_namesfiltered.xlsx'
        excel_path = path+'jira_issues_linked.xlsx'
    df.to_excel(excel_path, index=False, header=False, engine='openpyxl')

    print(f"Excel file has been created at: {excel_path}")

def getUniqueSets(path):
    # Step 1: Read the original Excel file
    original_excel_path = path
    df = pd.read_excel(original_excel_path)

    # Step 2: Build a graph of linked issues
    G = nx.Graph()
    for index, row in df.iterrows():
        issue_id = row[0]
        linked_issues = row[2]
        if pd.notna(linked_issues):
            for linked_issue in linked_issues.split(', '):
                G.add_edge(issue_id, linked_issue)

    # Step 3: Find connected components
    connected_components = list(nx.connected_components(G))

    # Step 4: Create the new DataFrame
    new_data = []
    for component in connected_components:
        new_data.append(['; '.join(component)])  # Joining each component's IDs into a single string

    new_df = pd.DataFrame(new_data, columns=['Unique Linked Issue Sets'])

    # Step 5: Write to Excel
    new_excel_path = 'data\issue_unique_sets.xlsx'  # Replace with your desired new Excel file path
    new_df.to_excel(new_excel_path, index=False)

    print(f"New Excel file has been created at: {new_excel_path}")

def applyFilters(summary,description, filterPrefix, filterSoftwareNames):
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
    return summary, description

def getLinkHierarchy(path, filterPrefix=False, filterSoftwareNames=None):
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
        type = issue.fields.issuetype.name
        summary = issue.fields.summary
        description = issue.fields.description
        if type == "User Subtask":
            print(key)
            # Fetch linked issues
            linked_sf = []
            linked_ws = []
            for link in issue.fields.issuelinks:
                if hasattr(link, "outwardIssue") and link.outwardIssue.fields.issuetype.name=="System Function":
                    linked_sf.append(link.outwardIssue.key)
                    sf = [x for x in issues if x.key == link.outwardIssue.key][0]
                    for sflink in sf.fields.issuelinks:
                        if hasattr(sflink, "inwardIssue") and sflink.inwardIssue.fields.issuetype.name=="Workspace":
                            linked_ws.append(sflink.inwardIssue.key)
                        elif hasattr(sflink, "outwardIssue") and sflink.outwardIssue.fields.issuetype.name=="Workspace":
                            linked_ws.append(sflink.outwardIssue.key)
                elif hasattr(link, "inwardIssue") and link.inwardIssue.fields.issuetype.name=="System Function":
                    linked_sf.append(link.inwardIssue.key)
                    sf = [x for x in issues if x.key == link.inwardIssue.key][0]
                    for sflink in sf.fields.issuelinks:
                        if hasattr(sflink, "inwardIssue") and sflink.inwardIssue.fields.issuetype.name=="Workspace":
                            linked_ws.append(sflink.inwardIssue.key)
                        elif hasattr(sflink, "outwardIssue") and sflink.outwardIssue.fields.issuetype.name=="Workspace":
                            linked_ws.append(sflink.outwardIssue.key)

            # Convert list of linked issues to a string
            print(linked_sf)
            print(linked_ws)
            linked_sf_issues_str = ', '.join(list(set(linked_sf)))
            linked_ws_issues_str = ', '.join(list(set(linked_ws)))
            data.append([key, linked_ws_issues_str, linked_sf_issues_str])

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Write DataFrame to Excel file
    if filterPrefix:
        excel_path = path+'jira_issues_linkhierarchy_noprefix.xlsx'
        if filterSoftwareNames != None:
            excel_path = path+'jira_issues_linkhierarchy_namesfiltered_noprefix.xlsx'
    else:
        if filterSoftwareNames != None:
            excel_path = path+'jira_issues_linkhierarchy_namesfiltered.xlsx'
        excel_path = path+'jira_issues_hierarchy.xlsx'
    df.to_excel(excel_path, index=False, header=False, engine='openpyxl')

    print(f"Excel file has been created at: {excel_path}")

extractIssuesFromJira("../data/sbert/", True, project="KOMOOTOLD")
#getLinkHierarchy("data/", True, ["Komoot", "Garmin", "Google Fit", "Strava"])
#getUniqueSets('data\jira_issues_linked_namesfiltered_noprefix.xlsx')