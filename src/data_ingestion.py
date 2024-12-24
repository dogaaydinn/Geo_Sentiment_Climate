import requests
import pandas as pd
import time


def fetch_openaq_data(api_key):
    """
    OpenAQ API'ye kimlik doğrulaması (Bearer Token) ile bağlanarak
    belirli parametreler ve tarih aralığında veri indirir.
    CSV dosyası olarak kaydeder.
    """

    base_url = "https://api.openaq.org/v2/measurements"

    # OpenAQ için gerekli "Authorization" header'ını ekle
    headers = {
        "Authorization": f"Bearer "
    }

    # Örnek olarak: ABD verisi (country=US),
    # parametreler (PM2.5, O3, NO2), belirli bir tarih aralığı, sayfalama limit=100
    params = {
        "country": "US",
        "parameter": ["pm25", "o3", "no2"],
        "date_from": "2021-01-01",
        "date_to": "2021-01-07",
        "limit": 100,
        "page": 1
    }

    all_data = []
    max_pages = 5  # Örnek: 5 sayfa (5 x 100 = 500 kayıt) alalım, gerekirse büyütebilirsin

    for page_num in range(1, max_pages + 1):
        params["page"] = page_num

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            json_data = response.json()
            results = json_data.get("results", [])
            all_data.extend(results)

            print(f"Sayfa {page_num} | Kayıt sayısı: {len(results)}")

            # Eğer bu sayfada kayıt yoksa, bir sonraki sayfada da olmayacaktır; döngüden çıkabiliriz
            if len(results) == 0:
                print("Veri kalmadı, döngüden çıkılıyor.")
                break
        else:
            print(f"HTTP Hatası: {response.status_code}")
            print(f"Yanıt metni: {response.text}")
            break

        # Rate limit veya istek sıklığına karşı ufak bekleme
        time.sleep(1)

    # Gelen tüm verileri DataFrame'e dönüştür
    df = pd.json_normalize(all_data)

    # CSV'ye kaydet - "data/raw" klasörüne kaydetmek istersen dizin yolunu değiştir
    output_file = "openaq_us_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Veri kaydedildi: {output_file}")


if __name__ == "__main__":
    # BURAYA KENDİ API KEY'İNİ YAZ
    MY_OPENAQ_API_KEY = ""

    fetch_openaq_data(MY_OPENAQ_API_KEY)
