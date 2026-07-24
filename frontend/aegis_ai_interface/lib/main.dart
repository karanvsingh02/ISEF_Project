import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(const AegisApp());

class AegisApp extends StatelessWidget {
  const AegisApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark(useMaterial3: true),
      home: const PinnTestScreen(),
    );
  }
}

class PinnTestScreen extends StatefulWidget {
  const PinnTestScreen({super.key});

  @override
  State<PinnTestScreen> createState() => _PinnTestScreenState();
}

class _PinnTestScreenState extends State<PinnTestScreen> {
  final _thicknessController = TextEditingController(text: "5.0");
  final _energyController = TextEditingController(text: "100.0");

  String _predictionResult = "Enter values and click predict.";
  bool _isLoading = false;

  Future<void> _runPinnPrediction() async {
    setState(() {
      _isLoading = true;
      _predictionResult = "Running PINN surrogate inference...";
    });

    final double? thickness = double.tryParse(_thicknessController.text);
    final double? energy = double.tryParse(_energyController.text);

    if (thickness == null || energy == null) {
      setState(() {
        _isLoading = false;
        _predictionResult = "Please enter valid numeric values.";
      });
      return;
    }

    try {
      final response = await http.post(
        Uri.parse('http://127.0.0.1:8000/api/v1/predict/pinn'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'thickness_cm': thickness,
          'energy_mev': energy,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _predictionResult =
              "Transmitted Dose: ${data['predicted_dose_mgy'].toStringAsFixed(4)} mGy\n"
              "Status: ${data['status']}";
        });
      } else {
        setState(() {
          _predictionResult = "Server Error: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _predictionResult = "Connection Failed: $e";
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AEGIS AI - PINN Surrogate Live Test')),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              controller: _thicknessController,
              decoration: const InputDecoration(
                labelText: 'Shield Thickness (cm)',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _energyController,
              decoration: const InputDecoration(
                labelText: 'Incident Proton Energy (MeV)',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: _isLoading ? null : _runPinnPrediction,
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(50),
              ),
              child: _isLoading
                  ? const CircularProgressIndicator()
                  : const Text('Predict Radiation Attenuation'),
            ),
            const SizedBox(height: 32),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey[900],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blueAccent),
              ),
              child: Text(
                _predictionResult,
                style: const TextStyle(fontSize: 16, fontFamily: 'monospace'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}