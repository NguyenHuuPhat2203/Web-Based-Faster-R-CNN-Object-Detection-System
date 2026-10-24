import { useEffect, useRef, useState } from "react";
import { fetchImageBlob } from "./api";

/**
 * Object URLs for protected images (Bearer auth cannot be attached to <img src>).
 * Fetches each id once, drops URLs for ids that leave the list, and revokes everything on unmount.
 */
export function useImageBlobUrls(imageIds: number[]): Record<number, string> {
  const [urls, setUrls] = useState<Record<number, string>>({});
  const requested = useRef(new Set<number>());
  const live = useRef(new Map<number, string>());
  const idsKey = imageIds.join(",");

  useEffect(() => {
    const wanted = new Set(imageIds);

    for (const id of requested.current) {
      if (wanted.has(id)) continue;
      requested.current.delete(id);
      const url = live.current.get(id);
      if (url) {
        URL.revokeObjectURL(url);
        live.current.delete(id);
        setUrls((prev) => {
          const { [id]: _removed, ...rest } = prev;
          return rest;
        });
      }
    }

    for (const id of wanted) {
      if (requested.current.has(id)) continue;
      requested.current.add(id);
      fetchImageBlob(id)
        .then((blob) => {
          // Pruned or unmounted while in flight, or a duplicate request won.
          if (!requested.current.has(id) || live.current.has(id)) return;
          const url = URL.createObjectURL(blob);
          live.current.set(id, url);
          setUrls((prev) => ({ ...prev, [id]: url }));
        })
        .catch(() => requested.current.delete(id));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey]);

  useEffect(() => {
    const liveUrls = live.current;
    const requestedIds = requested.current;
    return () => {
      liveUrls.forEach((url) => URL.revokeObjectURL(url));
      liveUrls.clear();
      requestedIds.clear();
    };
  }, []);

  return urls;
}
