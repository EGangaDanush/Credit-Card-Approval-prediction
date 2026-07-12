import pandas as pd
import numpy as np
def data_cleaning(applicant_df, credit_df):
    """
    Cleans and transforms applicant and credit datasets for credit approval prediction.
    Parameters:
        applicant_df (pd.DataFrame): Applicant dataset
        credit_df (pd.DataFrame): Credit records dataset
    Returns:
        pd.DataFrame: Cleaned applicant dataset
        pd.DataFrame: Transformed credit dataset
    """
    applicant_df['family_dependency'] = applicant_df['family_members'] + applicant_df['children']
    unnecessary_cols = ['phone_number', 'email']  
    applicant_df = applicant_df.drop(columns=[col for col in unnecessary_cols if col in applicant_df.columns])
    if 'DAYS_BIRTH' in applicant_df.columns:
        applicant_df['DAYS_BIRTH'] = applicant_df['DAYS_BIRTH'].abs()
    if 'DAYS_EMPLOYED' in applicant_df.columns:
        applicant_df['DAYS_EMPLOYED'] = applicant_df['DAYS_EMPLOYED'].abs()
    categorical_mappings = {
        'housing_type': {'own': 0, 'rent': 1, 'mortgage': 2, 'other': 3},
        'income_type': {'employed': 0, 'self-employed': 1, 'retired': 2, 'student': 3, 'other': 4},
        'education_type': {'secondary': 0, 'higher': 1, 'postgraduate': 2, 'other': 3},
        'family_type': {'single': 0, 'married': 1, 'divorced': 2, 'other': 3}
    }
    for col, mapping in categorical_mappings.items():
        if col in applicant_df.columns:
            applicant_df[col] = applicant_df[col].map(mapping)
    credit_grouped = credit_df.groupby('ID').agg({
        'MONTHS_BALANCE': ['min', 'max'],
        'STATUS': lambda x: list(x)
    }).reset_index()
    credit_grouped.columns = ['ID', 'open_month', 'end_months', 'status_list']
    credit_grouped['window'] = credit_grouped['end_months'] - credit_grouped['open_month']
    def interpret_status(status_list):
        if all(s == '0' for s in status_list):
            return 'timely'
        elif any(s in ['1', '2', '3', '4', '5'] for s in status_list):
            return 'overdue'
        elif all(s == 'C' for s in status_list):
            return 'closed'
        else:
            return 'no_record'
    credit_grouped['payment_behavior'] = credit_grouped['status_list'].apply(interpret_status)
    credit_grouped = credit_grouped.drop(columns=['status_list'])
    return applicant_df, credit_grouped