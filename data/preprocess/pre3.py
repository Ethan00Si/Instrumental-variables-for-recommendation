import pandas as pd
news = pd.read_csv('/Users/ethanxi/Desktop/mind/news_data.tsv', sep='\t')


'''gather texts into groups(query and item)'''

news_entity = news.copy()
news_entity['query'] = ''  # query text
news_entity['embedding']='' # item text 
news_entity = news_entity.drop(['NewsID','Category','SubCategory','Title','Abstract','Total_entities'],axis=1)
# news = news.to_numpy()
news = news.fillna('')

for idx in range(len(news)):
    # news_entity.loc[idx,'query'] = news.loc[idx,'Category']+' '+news.loc[idx,'SubCategory']+' '+news.loc[idx,'Total_entities']
    news_entity.loc[idx,'query'] = news.loc[idx,'SubCategory']+' '+news.loc[idx,'Total_entities']
    news_entity.loc[idx,'embedding'] = news.loc[idx,'Title']+' '+news.loc[idx,'Abstract']

news_entity.to_csv('./news_embedding_query.tsv',sep='\t',index=False)