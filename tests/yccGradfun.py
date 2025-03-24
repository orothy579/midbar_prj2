import ffmpeg

input_path = "/Users/lch/development/image_transformer/image/test_30_f1.jpg"
output_path = "/Users/lch/development/image_transformer/result/yccGradfun.png"

# 1. 입력 스트림 불러오기
input_stream = ffmpeg.input(input_path)

# 2. YUV 변환 후 Y 채널과 UV 채널 분리
split_streams = input_stream.filter(
    "format", "yuv444p").filter_multi_output("split")
y_channel = split_streams[0]
uv_channels = split_streams[1]

# 3. Y 채널에만 gradfun 적용
y_filtered = y_channel.filter("gradfun", 60, 20)

# 4. Y 채널과 UV 채널을 다시 병합
output_stream = ffmpeg.filter(
    [y_filtered, uv_channels], "mergeplanes", "0x001102", "yuv444p")

# 5. PNG 포맷으로 단일 이미지 저장
ffmpeg.output(output_stream, output_path, vcodec="png",
              format="image2", update=1).run(overwrite_output=True)

print(f"Debanded image saved to {output_path}")
