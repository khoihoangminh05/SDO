'use client';

import React, { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { getApiUrl } from '../store/explorationStore';

export default function Home() {
  const router = useRouter();
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const processFile = (selectedFile: File) => {
    // Basic frontend validation for safety
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(selectedFile.type)) {
      setError(
        'Invalid file type. Only JPEG, PNG, and WEBP images are allowed.',
      );
      setFile(null);
      return;
    }
    if (selectedFile.size > 50 * 1024 * 1024) {
      setError('File size too large. Maximum size is 50MB.');
      setFile(null);
      return;
    }
    setError(null);
    setFile(selectedFile);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch(getApiUrl('/upload'), {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.message || 'Failed to upload image');
      }

      const data = await res.json();
      // On success, redirect to the viewer page passing the filename
      router.push(`/viewer/${data.filename}`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Something went wrong during upload',
      );
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col justify-center items-center p-6 font-sans">
      <div className="w-full max-w-lg bg-slate-800/40 p-8 rounded-2xl border border-slate-800 shadow-2xl flex flex-col items-center">
        <h1 className="text-3xl font-extrabold mb-2 text-center bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
          Recursive Object Detector
        </h1>
        <p className="text-sm text-slate-400 mb-8 text-center">
          Upload high-resolution images to run YOLO detection tree overlays
        </p>

        <form
          onSubmit={handleSubmit}
          className="w-full flex flex-col items-center"
        >
          <div
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            className={`relative w-full h-64 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-6 transition-all duration-300 ${
              dragActive
                ? 'border-blue-500 bg-blue-500/5'
                : file
                  ? 'border-emerald-500 bg-emerald-500/5'
                  : 'border-slate-700 bg-slate-850 hover:border-slate-600'
            }`}
          >
            <input
              type="file"
              id="file-upload"
              accept="image/jpeg, image/png, image/webp"
              onChange={handleChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />

            {!file ? (
              <div className="flex flex-col items-center pointer-events-none text-slate-400">
                <svg
                  className="w-12 h-12 mb-4 text-slate-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  ></path>
                </svg>
                <p className="text-sm font-semibold mb-1">
                  Drag and drop your image here
                </p>
                <p className="text-xs text-slate-500">
                  JPEG, PNG, or WEBP up to 50MB
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center pointer-events-none text-center">
                <svg
                  className="w-12 h-12 mb-4 text-emerald-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  ></path>
                </svg>
                <p className="text-sm font-semibold text-emerald-400 truncate max-w-xs mb-1">
                  {file.name}
                </p>
                <p className="text-xs text-slate-500">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 text-xs text-red-400 bg-red-500/10 border border-red-500/20 px-3 py-2 rounded-lg w-full text-center">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={!file || uploading}
            className={`mt-6 w-full py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg ${
              !file || uploading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                : 'bg-blue-600 hover:bg-blue-500 hover:shadow-blue-500/20 text-white cursor-pointer'
            }`}
          >
            {uploading ? (
              <div className="flex items-center justify-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Uploading...</span>
              </div>
            ) : (
              'Analyze Image'
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
