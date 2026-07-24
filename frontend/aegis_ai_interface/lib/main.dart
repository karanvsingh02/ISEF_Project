import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: BackendTestScreen(),
    );
  }
}

class BackendTestScreen extends StatefulWidget {
  const BackendTestScreen({super.key});

  @override
  State<BackendTestScreen> createState() => _BackendTestScreenState();
}

class _BackendTestScreenState extends State<BackendTestScreen> {
  String _apiStatus = "Connecting to FastAPI...";

  @override
  void initState() {
    super.initState();
    _checkBackendHealth();
  }

  Future<void> _checkBackendHealth() async {
    try {
      final response = await http.get(Uri.parse('http://127.0.0.1:8000/'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _apiStatus = "SUCCESS: Connected to Backend!\n\nMessage: ${data['message']}";
        });
      } else {
        setState(() {
          _apiStatus = "Error: Status Code ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _apiStatus = "Connection failed. Is Uvicorn running on port 8000?\nError: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AEGIS AI - Backend Connection Test')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Text(
            _apiStatus,
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }
}