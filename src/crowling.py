import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

def main(hashtag_list):
    name = hashtag_list[0]
    hashtag = str(hashtag_list[1])

    webdriver_options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=webdriver_options)

    driver.get(f'https://m.place.naver.com/restaurant/{hashtag}/review/visitor?entry=ple&reviewSort=recent')
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)

    try:
        while True:
            time.sleep(5)
            button = driver.find_element(By.XPATH, '//*[@id="app-root"]/div/div/div/div[6]/div[2]/div[3]/div[2]/div/a')
            button.click()
            print("더보기 버튼 클릭 완료")
    except:
        print("더 이상 더보기 버튼이 없습니다. 모든 데이터를 로드했습니다.")

    reviews = driver.find_elements(By.CSS_SELECTOR, 'li.pui__X35jYm.EjjAW')
    df = pd.DataFrame(columns=['nickname', 'content', 'date', 'revisit'])

    for r in reviews:
        nickname_element = r.find_element(By.CSS_SELECTOR, 'div.pui__JiVbY3 > span.pui__uslU0d')
        nickname = nickname_element.text if nickname_element else ''

        content_element = r.find_element(By.CSS_SELECTOR, 'div.pui__vn15t2 > a.pui__xtsQN-')
        content = content_element.text if content_element else ''

        date_elements = r.find_elements(By.CSS_SELECTOR, 'div.pui__QKE5Pr > span.pui__gfuUIT > time')
        date = date_elements[0].text if date_elements else 'N/A'

        revisit_span = r.find_elements(By.CSS_SELECTOR, 'div.pui__QKE5Pr > span.pui__gfuUIT')
        revisit = revisit_span[1].text if len(revisit_span) > 1 else 'N/A'

        new_row = pd.DataFrame({
            'nickname': [nickname],
            'content': [content],
            'date': [date],
            'revisit': [revisit]
        })
        df = pd.concat([df, new_row], ignore_index=True)

    print(f"총 데이터 수: {len(reviews)}")
    print(f'{name} 리뷰 데이터 수집 완료.')
    df.to_excel(f'./{name}_리뷰.xlsx', index=False)
    driver.quit()
    return df

if __name__ == "__main__":
    hashtag_lists = [
        ['유자유김치떡볶이_신촌점', 1197641297],
        ['크리스터_치킨', 32528373],
        ['빠빠빠치킨_연대본점', 1733132588],
        ['에일크루브루잉_신촌점', 1027901503],
        ['목구멍_신촌점', 1682499258],
        ['하나마토_신촌점', 1161590191],
        ['착한곱창', 1436458587],
        ['오향미엔', 1681222394],
        ['홍미닭발', 35458332],
        ['김덕후의곱창조_신촌점', 37157148],
        ['이자카야_우규_신촌점', 1433153629],
        ['고삼이_신촌점', 20601663],
        ['신촌정직한족발', 1913676499],
        ['더도이축산직영점_신촌점', 1910033423]
    ]

    for h in hashtag_lists:
        main(h)
        time.sleep(5)
