# https://data-newbie.tistory.com/205

## gather 함수 
def gather( df, key, value, cols ):
    id_vars = [ col for col in df.columns if col not in cols ]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt( df, id_vars, id_values, var_name, value_name )
    
gather( df, 'drug', 'heartrate', ['a','b'] )
