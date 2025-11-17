    def _download_result(self, process_id, api_key, output_format):
        accept = FORMAT_ACCEPT.get(output_format, "image/jpeg")
        headers = {"X-API-Key": api_key, "Accept": accept}
        download_url = f"{BASE_URL}/download/{process_id}"
        print(f"[Topaz] Download URL: {download_url}")

        # Retry up to 5 times with small delay (download often lags 1-2s behind status)
        for attempt in range(5):
            resp = requests.get(download_url, headers=headers, timeout=30)
            print(f"[Topaz] Download attempt {attempt + 1} → code: {resp.status_code}")

            # If we get HTML or JSON → not an image
            if resp.status_code == 200:
                # Quick sanity check: real JPEG starts with b'\xff\xd8\xff'
                if resp.content.startswith(b'\xff\xd8\xff'):
                    print(f"[Topaz] Valid JPEG received ({len(resp.content)} bytes)")
                    return resp.content
                else:
                    # Got 200 but garbage → likely error page
                    preview = resp.content[:200].decode('utf-8', errors='ignore')
                    print(f"[Topaz] Invalid content (not JPEG): {preview}")

            elif resp.status_code == 404:
                print("[Topaz] 404 - download not ready yet, waiting...")
            else:
                print(f"[Topaz] HTTP {resp.status_code}: {resp.text[:200]}")

            if attempt < 4:
                time.sleep(3)  # Wait and retry
        raise Exception(f"Failed to download valid image after 5 attempts. Process ID: {process_id}")
