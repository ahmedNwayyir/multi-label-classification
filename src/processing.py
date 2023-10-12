import numpy as np
import polars as pl
import pandas as pd
import spacy

import src.utils as ut


nlp_small = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp_large = spacy.load("en_core_web_lg", disable=["parser", "ner"])

def text_tokenizer(text: str, nlp: spacy.lang.en.English=nlp_small) -> list:
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.lemma_.isalpha() and not token.is_stop]


def tokenize_sof_df(file_path: str, process_polars: bool=False) -> pl.DataFrame:
    df = pd.read_csv(file_path)
    
    if process_polars:
        return (
            pl.from_pandas(df)  # polars has problems reading data from the csv file
            .pipe(ut.get_df_info, msg='Before processing:')
            .select(
                    pl.col('Id').cast(pl.UInt32).alias('id'),
                    pl.col('Title').cast(pl.Utf8).map_elements(text_tokenizer).alias('title'),
                    pl.col('Body').cast(pl.Utf8).map_elements(text_tokenizer).alias('body'),
                    pl.col('Tags').cast(pl.Utf8).map_elements(lambda x: x[1:-1].split('><')).alias('tags'),
                    pl.col('Y').cast(pl.Categorical).alias('y'),
                )
            .pipe(ut.get_df_info, msg='\nAfter processing:')
        )
        
    else:
        return (
        df
        .pipe(ut.get_df_info, msg='Before processing:')
        .assign(
            id = lambda _df: _df['Id'].astype(np.uint32),
            title = lambda _df: _df['Title'].astype(pd.StringDtype()).apply(text_tokenizer),
            body = lambda _df: _df['Body'].astype(pd.StringDtype()).apply(text_tokenizer),
            tags = lambda _df: _df['Tags'].astype(pd.StringDtype()).apply(lambda x: x[1:-1].split('><')),
            y = lambda _df: _df['Y'].astype('category'),
        )
        .loc[:, ['id', 'title', 'body', 'tags', 'y']]
        .pipe(ut.get_df_info, msg='\nAfter processing:')
    )


def get_counts(df: pl.DataFrame) -> pl.DataFrame:
    if type(df) == pl.DataFrame:
        stats = (
            pl.DataFrame()
            .with_columns(
                title_count = df['title'].map_elements(len),
                body_count = df['body'].map_elements(len),
                tags_count = df['tags'].map_elements(len)
            )
        )
        print(stats.describe())
        return stats
    elif type(df) == pd.DataFrame:
        stats = (
            pd.DataFrame()
            .assign(
                title_count = df['title'].map(len),
                body_count = df['body'].map(len),
                tags_count = df['tags'].map(len)
            )
        )
        print(stats.describe(include='all'))
        return stats