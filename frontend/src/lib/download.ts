/** Trigger a browser download for a blob and release its object URL. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  // Revoke after the click has been dispatched; the download keeps its own reference.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
