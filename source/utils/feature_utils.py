from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(df, cols, method="standard"):

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    df[cols] = scaler.fit_transform(df[cols])
    return df

def create_interaction_terms(df, col1, col2, operation="multiply"):

    if operation == "multiply":
        return df[col1] * df[col2]
    elif operation == "add":
        return df[col1] + df[col2]
    else:
        raise ValueError("operation must be 'multiply' or 'add'")
