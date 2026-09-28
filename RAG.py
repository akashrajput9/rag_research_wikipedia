#  pa-NBbmyhielEIxcG-th92ZUPL3hqDM_Aa7yEELF4UEv-y


import voyageai

vo = voyageai.Client("pa-NBbmyhielEIxcG-th92ZUPL3hqDM_Aa7yEELF4UEv-y")
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

result = vo.embed(["hello world"], model="voyage-4-large")

print(result)